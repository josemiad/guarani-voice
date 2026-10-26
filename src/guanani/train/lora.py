"""Fine-tuning LoRA de NLLB, selección por chrF++ en FLORES+ dev y fusión del adaptador."""

import gc
import json
import re
import shutil
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

from guanani import data
from guanani.eval import score_mt
from guanani.nmt import MADLAD_PREFIX, family_of, load_seq2seq
from guanani.normalize import normalize
from guanani.train.corpus import build_examples
from guanani.train.sources import allowed_sources, read_jsonl, train_dir

BASE_MODEL = "facebook/nllb-200-distilled-1.3B"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
# T5 (MADLAD, gated-gelu): atención q/k/v/o y FFN wi_0/wi_1/wo.
T5_TARGET_MODULES = ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]
MAX_LEN = 200
CODES = {"gn": "grn_Latn", "es": "spa_Latn"}
_MADLAD_CODE = {"grn_Latn": "gn", "spa_Latn": "es"}
_BT_KIND = re.compile(r"bt\d+")
# Modelos con licencia comercial aceptados para generar la retrotraducción (el español sintético).
COMMERCIAL_BT_MODELS = ("google/madlad400-",)


def lora_targets(family: str) -> list[str]:
    return T5_TARGET_MODULES if family == "madlad" else TARGET_MODULES


def tokenize_examples(tok, examples: list[dict], max_len: int = MAX_LEN, family: str = "nllb") -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for ex in examples:
        groups.setdefault((ex["src_lang"], ex["tgt_lang"]), []).append(ex)
    out = []
    for (src_lang, tgt_lang), exs in groups.items():
        srcs = [e["src"] for e in exs]
        if family == "madlad":
            srcs = [MADLAD_PREFIX[_MADLAD_CODE[tgt_lang]] + s for s in srcs]
        else:
            tok.src_lang, tok.tgt_lang = src_lang, tgt_lang
        enc = tok(srcs, text_target=[e["tgt"] for e in exs], max_length=max_len, truncation=True)
        out += [
            {"input_ids": i, "attention_mask": a, "labels": lab}
            for i, a, lab in zip(enc["input_ids"], enc["attention_mask"], enc["labels"])
        ]
    return out


def dev_chrf(model, tok, pairs: list[dict], direction: str, batch_size: int = 16) -> float:
    src, tgt = direction.split("-")
    srcs = [p[src] for p in pairs]
    gen = {"num_beams": 1}
    if family_of(model.config) == "madlad":
        srcs = [MADLAD_PREFIX[tgt] + s for s in srcs]
    else:
        tok.src_lang = CODES[src]
        gen["forced_bos_token_id"] = tok.convert_tokens_to_ids(CODES[tgt])
    was_training = model.training
    model.eval()
    hyps = []
    for i in range(0, len(srcs), batch_size):
        enc = tok(srcs[i : i + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN).to(model.device)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=MAX_LEN, **gen)
        hyps += tok.batch_decode(out, skip_special_tokens=True)
    if was_training:
        model.train()
    refs = [p[tgt] for p in pairs]
    if tgt == "gn":
        hyps, refs = [normalize(h) for h in hyps], [normalize(r) for r in refs]
    return score_mt(hyps, refs)["chrf++"]


DIRECTIONS = ("gn-es", "es-gn")


class DevChrF(TrainerCallback):
    def __init__(self, tok, dev_pairs: list[dict], eval_steps: int, out_dir: Path, best_name: str = "best_adapter",
                 per_direction: bool = False):
        self.tok, self.dev, self.eval_steps, self.out_dir = tok, dev_pairs, eval_steps, out_dir
        self.best_dir, self.per_direction = out_dir / best_name, per_direction
        self.path = out_dir / "dev_history.json"
        self.history = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.best, self.best_step = float("-inf"), None
        self.dir_best = dict.fromkeys(DIRECTIONS, float("-inf"))
        self.dir_best_step = dict.fromkeys(DIRECTIONS)
        for step, s in sorted(self.history.items(), key=lambda kv: int(kv[0])):
            self._track(s, int(step))

    def _track(self, scores: dict, step: int) -> list[Path]:
        """Actualiza los mejores y devuelve dónde guardar el modelo de este paso (si mejora alguno)."""
        targets = []
        if scores["mean"] > self.best:
            self.best, self.best_step = scores["mean"], step
            if not self.per_direction:
                targets.append(self.best_dir)
        for d in DIRECTIONS:
            if scores[d] > self.dir_best[d]:
                self.dir_best[d], self.dir_best_step[d] = scores[d], step
                if self.per_direction:
                    targets.append(self.out_dir / f"{self.best_dir.name}_{d}")
        return targets

    def _evaluate(self, model, step: int) -> None:
        scores = {d: dev_chrf(model, self.tok, self.dev, d) for d in DIRECTIONS}
        scores["mean"] = round((scores["gn-es"] + scores["es-gn"]) / 2, 2)
        self.history[str(step)] = scores
        for target in self._track(scores, step):
            model.save_pretrained(target)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        best = self.dir_best_step if self.per_direction else f"{self.best} en {self.best_step}"
        print(f"[dev] paso {step}: {scores} (mejor {best})", flush=True)

    def on_train_begin(self, args, state, control, model=None, **kw):
        # Paso 0 = modelo base (el LoRA recién creado no lo modifica): así puede ganar la base.
        if "0" not in self.history:
            self._evaluate(model, 0)

    def on_step_end(self, args, state, control, model=None, **kw):
        if state.global_step % self.eval_steps == 0:
            self._evaluate(model, state.global_step)

    def on_train_end(self, args, state, control, model=None, **kw):
        if str(state.global_step) not in self.history:
            self._evaluate(model, state.global_step)


def prepare_out_dir(out: Path, resume: bool) -> None:
    # Una carrera nueva no debe heredar el historial ni el mejor adaptador de otra anterior con el mismo nombre.
    if resume:
        return
    (out / "dev_history.json").unlink(missing_ok=True)
    for sub in ("best_adapter", "best_adapter_gn-es", "best_adapter_es-gn", "best_model", "checkpoints"):
        shutil.rmtree(out / sub, ignore_errors=True)


def train(
    name: str,
    examples: list[dict],
    dev_pairs: list[dict],
    base: str = BASE_MODEL,
    epochs: float = 2.0,
    batch_size: int = 8,
    grad_accum: int = 4,
    lr: float = 2e-4,
    max_steps: int = -1,
    eval_steps: int = 1000,
    resume: bool = False,
    full: bool = False,
    per_direction: bool = False,
    models_dir: Path = Path("models"),
) -> dict:
    out = models_dir / name
    prepare_out_dir(out, resume)
    tok = AutoTokenizer.from_pretrained(base)
    model = load_seq2seq(base, torch.bfloat16)
    family = family_of(model.config)
    if not full:
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                 target_modules=lora_targets(family), task_type="SEQ_2_SEQ_LM"))
        model.print_trainable_parameters()
    cb = DevChrF(tok, dev_pairs, eval_steps, out, best_name="best_model" if full else "best_adapter",
                per_direction=per_direction)
    args = Seq2SeqTrainingArguments(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        optim="adafactor" if full else "adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=epochs,
        max_steps=max_steps,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=50,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        report_to=[],
        seed=42,
        remove_unused_columns=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenize_examples(tok, examples, family=family),
        data_collator=DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100),
        processing_class=tok,
        callbacks=[cb],
    )
    oom = False
    try:
        trainer.train(resume_from_checkpoint=True if resume else None)
    except torch.OutOfMemoryError:
        oom = True
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    if oom:
        raise RuntimeError(
            f"Sin VRAM con --batch-size {batch_size}: prueba --batch-size {max(1, batch_size // 2)} "
            f"--grad-accum {grad_accum * 2} (mismo batch efectivo)"
        )
    return {"best_step": cb.best_step, "best_dev": cb.history.get(str(cb.best_step)), "history": cb.history,
            "best_by_direction": {d: {"step": cb.dir_best_step[d], "chrf++": cb.dir_best[d]} for d in DIRECTIONS}}


def merge(base: str, adapter_dir: Path, out_dir: Path) -> None:
    tok = AutoTokenizer.from_pretrained(base)
    m = load_seq2seq(base, torch.float16)
    m = PeftModel.from_pretrained(m, str(adapter_dir)).merge_and_unload()
    m.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)


def finalize_full(out_dir: Path, base: str) -> None:
    best = out_dir / "best_model"
    for f in best.iterdir():
        shutil.move(str(f), out_dir / f.name)
    best.rmdir()
    AutoTokenizer.from_pretrained(base).save_pretrained(out_dir)


def data_file(kind: str) -> tuple[Path, str]:
    if kind == "parallel":
        return train_dir() / "parallel.jsonl", "train"
    if kind == "bt":
        return train_dir() / "bt.jsonl", "bt"
    if _BT_KIND.fullmatch(kind):
        return train_dir() / kind / f"{kind}.jsonl", "bt"
    raise ValueError(f"--data desconocido: {kind}; opciones: parallel, bt, btN (p. ej. bt2)")


def load_examples(
    data_kinds: list[str], upsample_parallel: int = 1, bt_tag: bool = False, allow: set[str] | None = None
) -> tuple[list[dict], dict]:
    paths = {k: data._require(*data_file(k)) for k in data_kinds}
    rows = {k: read_jsonl(p) for k, p in paths.items()}
    if allow is not None:
        for k, p in paths.items():
            if k == "parallel":
                continue
            meta = p.parent / "meta.json"
            model = json.loads(meta.read_text(encoding="utf-8")).get("model", "") if k != "bt" and meta.exists() else ""
            if not model.startswith(COMMERCIAL_BT_MODELS):
                raise ValueError(f"{k}: retrotraducción generada con {model or 'un modelo desconocido'}, "
                                 f"no con licencia comercial ({', '.join(COMMERCIAL_BT_MODELS)}*)")
        rows = {k: [r for r in v if r["source"] in allow] for k, v in rows.items()}
        empty = [k for k, v in rows.items() if not v]
        if empty:
            raise ValueError(f"Sin ejemplos con licencia aprobada en: {empty}")
    parallel = rows.get("parallel", []) * upsample_parallel
    bt = [r for k, v in rows.items() if k != "parallel" for r in v]
    examples = build_examples(parallel, bt, bt_tag=bt_tag)
    return examples, {"n_parallel": len(parallel), "n_bt": len(bt), "n_examples": len(examples)}


def run(
    name: str,
    data_kinds: list[str],
    dev_size: int = 300,
    models_dir: Path = Path("models"),
    upsample_parallel: int = 1,
    bt_tag: bool = False,
    sources_allow: str | None = None,
    **train_kwargs,
) -> Path:
    if train_kwargs.get("full") and train_kwargs.get("per_direction"):
        raise ValueError("--per-direction solo está soportado con LoRA, no con --full")
    allow = allowed_sources(Path(sources_allow)) if sources_allow else None
    examples, counts = load_examples(data_kinds, upsample_parallel, bt_tag, allow)
    dev = data.load_flores("dev")[:dev_size]
    base = train_kwargs.get("base", BASE_MODEL)
    res = train(name, examples, dev, models_dir=models_dir, **train_kwargs)
    out = models_dir / name
    out.mkdir(parents=True, exist_ok=True)
    if train_kwargs.get("per_direction"):
        for d in DIRECTIONS:
            merge(base, out / f"best_adapter_{d}", out / d)
    elif train_kwargs.get("full"):
        finalize_full(out, base)
    else:
        merge(base, out / "best_adapter", out)
    meta = {"name": name, "base": base, "data": data_kinds, "upsample_parallel": upsample_parallel,
            "bt_tag": bt_tag, "sources_allow": sources_allow,
            "allowed_sources": sorted(allow) if allow else None, "dev_size": dev_size, "train_kwargs": train_kwargs} | counts | res
    (out / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
