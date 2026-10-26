"""Afinado completo de Whisper para guaraní (etiqueta de idioma es), mejor punto por CER en Common Voice dev."""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import TrainerCallback

from guanani.stt import SR, WHISPER_LANG, load_audio

BASE = "openai/whisper-large-v3-turbo"
LANG_NAME = "spanish"  # nombre que espera WhisperProcessor para WHISPER_LANG


def check_base(base: str) -> None:
    if not base.startswith("openai/whisper-"):
        raise ValueError(f"modelo base no permitido para uso comercial: {base} (solo openai/whisper-*)")


def usable_clips(rows: list[dict], max_s: float = 30.0) -> tuple[list[dict], dict]:
    # ponytail: decodifica cada clip una vez para validarlo; con other (~19k clips) tarda unos minutos.
    stats = dict.fromkeys(("in", "largo", "ilegible", "aceptado"), 0)
    kept = []
    for r in rows:
        stats["in"] += 1
        try:
            n = len(load_audio(r["audio"]))
        except Exception:
            stats["ilegible"] += 1
            continue
        if n > max_s * SR:
            stats["largo"] += 1
        else:
            stats["aceptado"] += 1
            kept.append(r)
    return kept, stats


class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, rows, proc):
        self.rows, self.proc = rows, proc

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        feats = self.proc.feature_extractor(load_audio(r["audio"]), sampling_rate=SR).input_features[0]
        return {"input_features": feats, "labels": self.proc.tokenizer(r["text"]).input_ids}


class Collator:
    def __init__(self, proc):
        self.proc = proc
        self.sot = proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

    def __call__(self, feats):
        batch = self.proc.feature_extractor.pad([{"input_features": f["input_features"]} for f in feats], return_tensors="pt")
        lab = self.proc.tokenizer.pad([{"input_ids": f["labels"]} for f in feats], return_tensors="pt")
        labels = lab["input_ids"].masked_fill(lab["attention_mask"].ne(1), -100)
        if (labels[:, 0] == self.sot).all():  # el modelo antepone el token de inicio por su cuenta
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


class DevCER(TrainerCallback):
    """Evalúa CER en dev cada eval_steps (y en el paso 0) y guarda el mejor modelo. Menor es mejor."""

    def __init__(self, proc, dev_rows: list[dict], eval_steps: int, out_dir: Path):
        self.proc, self.dev, self.eval_steps, self.out_dir = proc, dev_rows, eval_steps, out_dir
        self.path = out_dir / "dev_history.json"
        self.history = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.best, self.best_step = float("inf"), None
        for step, cer in self.history.items():
            if cer < self.best:
                self.best, self.best_step = cer, int(step)

    def _cer(self, model) -> float:
        from guanani.eval import score_asr

        was_training = model.training
        model.eval()
        hyps = []
        for i in range(0, len(self.dev), 16):
            audios = [load_audio(r["audio"]) for r in self.dev[i:i + 16]]
            feats = self.proc(audios, sampling_rate=SR, return_tensors="pt").input_features.to(model.device, model.dtype)
            with torch.inference_mode(), torch.autocast(model.device.type, dtype=torch.bfloat16, enabled=model.device.type == "cuda"):
                ids = model.generate(feats, language=WHISPER_LANG, task="transcribe", num_beams=1, max_new_tokens=225)
            hyps += self.proc.batch_decode(ids, skip_special_tokens=True)
        model.train(was_training)
        return score_asr(hyps, [r["text"] for r in self.dev])["cer"]

    def _evaluate(self, model, step: int) -> None:
        cer = self._cer(model)
        self.history[str(step)] = cer
        if cer < self.best:
            self.best, self.best_step = cer, step
            model.save_pretrained(self.out_dir / "best_model")
            self.proc.save_pretrained(self.out_dir / "best_model")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        print(f"[dev] paso {step}: CER {cer} (mejor {self.best} en {self.best_step})", flush=True)

    def on_train_begin(self, args, state, control, model=None, **kw):
        if "0" not in self.history:
            self._evaluate(model, 0)

    def on_step_end(self, args, state, control, model=None, **kw):
        if state.global_step % self.eval_steps == 0:
            self._evaluate(model, state.global_step)

    def on_train_end(self, args, state, control, model=None, **kw):
        if str(state.global_step) not in self.history:
            self._evaluate(model, state.global_step)


def run(name: str, data_kinds=("train",), base: str = BASE, epochs: float = 20.0, batch_size: int = 16,
        grad_accum: int = 2, lr: float = 1e-5, max_steps: int = -1, eval_steps: int = 100, resume: bool = False,
        models_dir: Path = Path("models"), dev_limit: int | None = None) -> dict:
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor

    from guanani.train import stt_data
    from guanani.train.lora import prepare_out_dir

    check_base(base)
    rows, filt = stt_data.load_stt(list(data_kinds))
    dev, _ = stt_data.load_stt(["dev"])
    train_rows, clip_stats = usable_clips(rows)
    dev_rows, _ = usable_clips(dev[:dev_limit])
    out = models_dir / name
    out.mkdir(parents=True, exist_ok=True)
    prepare_out_dir(out, resume)

    proc = WhisperProcessor.from_pretrained(base, language=LANG_NAME, task="transcribe")
    # fp32 + autocast bf16: en bf16 puro se perdían las actualizaciones pequeñas (subproyecto 1c).
    model = WhisperForConditionalGeneration.from_pretrained(base)
    model.config.apply_spec_augment, model.config.mask_time_prob = True, 0.05
    model.generation_config.language, model.generation_config.task = WHISPER_LANG, "transcribe"
    model.generation_config.forced_decoder_ids = None

    cuda = torch.cuda.is_available()
    cb = DevCER(proc, dev_rows, eval_steps, out)
    args = Seq2SeqTrainingArguments(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_steps=50,
        num_train_epochs=epochs,
        max_steps=max_steps,
        bf16=cuda,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        logging_steps=10,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=4 if cuda else 0,
    )
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=ClipDataset(train_rows, proc),
                             data_collator=Collator(proc), callbacks=[cb])
    trainer.train(resume_from_checkpoint=True if resume else None)
    meta = {"base": base, "data": list(data_kinds), "filtrado": filt, "clips": clip_stats, "dev": len(dev_rows),
            "best_step": cb.best_step, "best_cer": cb.best, "history": cb.history}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta
