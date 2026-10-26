"""CLI: guanani translate|transcribe|speak|pipeline|eval."""

import argparse
import json
import sys

from guanani.data import DATASETS, DataMissing


def _kw(a) -> dict:
    return {"model": a.model} if getattr(a, "model", None) else {}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="guanani")
    p.add_argument("--device", choices=["cuda", "cpu"])
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("translate", help="traduce texto gn↔es")
    t.add_argument("text")
    t.add_argument("--src", choices=["gn", "es"], default="gn")
    t.add_argument("--tgt", choices=["gn", "es"], default="es")
    t.add_argument("--model")

    s = sub.add_parser("transcribe", help="voz guaraní → texto")
    s.add_argument("audio")
    s.add_argument("--model")
    s.add_argument("--long", action="store_true", help="trocea por silencios aunque el audio sea corto")

    k = sub.add_parser("speak", help="texto guaraní → voz")
    k.add_argument("text")
    k.add_argument("-o", "--out", default="out.wav")
    k.add_argument("--model")

    pl = sub.add_parser("pipeline", help="voz gn → texto es, o texto es → voz gn")
    g = pl.add_mutually_exclusive_group(required=True)
    g.add_argument("--audio", help="audio en guaraní → traducción al español")
    g.add_argument("--text", help="texto en español → voz en guaraní")
    pl.add_argument("-o", "--out", default="out.wav")

    e = sub.add_parser("eval", help="evalúa un componente y escribe el informe en results/")
    e.add_argument("component", choices=["nmt", "stt", "tts"])
    e.add_argument("dataset", choices=list(DATASETS))
    e.add_argument("--direction", choices=["gn-es", "es-gn"], default="gn-es")
    e.add_argument("--model")
    e.add_argument("--limit", type=int)
    e.add_argument("--no-comet", action="store_true")
    e.add_argument("--asr-model", help="reconocedor de ida y vuelta para tts")
    e.add_argument("--no-utmos", action="store_true")

    tr = sub.add_parser("train", help="prepara datos y entrena LoRA de NLLB")
    tsub = tr.add_subparsers(dest="train_cmd", required=True)
    tsub.add_parser("prepare", help="construye data/train/parallel.jsonl")
    b = tsub.add_parser("backtranslate", help="retrotraduce frases de KuatIA a data/train/bt.jsonl")
    b.add_argument("--name", required=True, help="carpeta en data/train/ (p. ej. bt2)")
    b.add_argument("--n", type=int, default=500000)
    b.add_argument("--model")
    b.add_argument("--sample", action="store_true", help="muestreo top-k en vez de beam")
    b.add_argument("--batch-size", type=int, default=16)
    b.add_argument("--no-lid", action="store_true", help="sin filtro GlotLID")
    b.add_argument("--from", dest="from_", help="reutiliza la muestra de data/train/<from>/")
    lo = tsub.add_parser("nmt", aliases=["lora"], help="entrena NLLB/MADLAD (LoRA o --full) y guarda en models/<name>")
    lo.add_argument("--full", action="store_true", help="fine-tuning completo (Adafactor) en vez de LoRA")
    lo.add_argument("--per-direction", action="store_true", help="mejor checkpoint por dirección: models/<name>/gn-es y es-gn")
    lo.add_argument("--sources-allow", help="tabla de licencias: solo fuentes con Commercial = yes")
    lo.add_argument("--name", required=True)
    lo.add_argument("--data", default="parallel", help="parallel o parallel,bt")
    lo.add_argument("--epochs", type=float, default=2.0)
    lo.add_argument("--batch-size", type=int, default=8)
    lo.add_argument("--grad-accum", type=int, default=4)
    lo.add_argument("--lr", type=float, default=2e-4)
    lo.add_argument("--max-steps", type=int, default=-1)
    lo.add_argument("--eval-steps", type=int, default=1000)
    lo.add_argument("--dev-size", type=int, default=300)
    lo.add_argument("--base", default="facebook/nllb-200-distilled-1.3B")
    lo.add_argument("--resume", action="store_true")
    lo.add_argument("--upsample-parallel", type=int, default=1, help="repite los pares reales N veces")
    lo.add_argument("--bt-tag", action="store_true", help="antepone [BT] al español retrotraducido")

    oe = tsub.add_parser("omni-export", help="exporta Common Voice al parquet de la receta de Omnilingual")
    oe.add_argument("--name", required=True)
    oe.add_argument("--data", default="train", help="train o train,other")
    op = tsub.add_parser("omni-pick", help="elige el mejor checkpoint de Omnilingual por CER en cv dev")
    op.add_argument("--name", required=True)
    op.add_argument("--run-dir", required=True)
    op.add_argument("--arch", choices=["300m_v2", "1b_v2", "3b_v2"], required=True)
    op.add_argument("--dev-limit", type=int)

    st = tsub.add_parser("stt", help="afina Whisper (models/<name>/best_model)")
    st.add_argument("--name", required=True)
    st.add_argument("--data", default="train", help="train o train,other")
    st.add_argument("--base", default="openai/whisper-large-v3-turbo")
    st.add_argument("--epochs", type=float, default=20.0)
    st.add_argument("--batch-size", type=int, default=16)
    st.add_argument("--grad-accum", type=int, default=2)
    st.add_argument("--lr", type=float, default=1e-5)
    st.add_argument("--max-steps", type=int, default=-1)
    st.add_argument("--eval-steps", type=int, default=100)
    st.add_argument("--dev-limit", type=int)
    st.add_argument("--resume", action="store_true")

    tp = tsub.add_parser("tts-prep", help="prepara la voz de un hablante de Common Voice para Piper")
    tp.add_argument("--name", required=True)
    tp.add_argument("--speaker", help="client_id; por defecto el hablante con más clips")
    tp.add_argument("--asr-model", default="models/w2")
    tp.add_argument("--max-cer", type=float, default=0.10)

    tk = tsub.add_parser("tts-pick", help="exporta los checkpoints de Piper y deja el mejor en models/<name>")
    tk.add_argument("--name", required=True)
    tk.add_argument("--run-dir", required=True)
    tk.add_argument("--asr-model", default="models/w2")
    tk.add_argument("--dev-limit", type=int, default=100)

    a = p.parse_args(argv)
    try:
        if a.cmd == "translate":
            from guanani.nmt import translate

            print(translate([a.text], a.src, a.tgt, device=a.device, **_kw(a))[0])
        elif a.cmd == "transcribe":
            from guanani import stt

            audio = stt.load_audio(a.audio)
            # Más largo que un fragmento: se trocea siempre (Whisper cortaría a 30 s, Omnilingual falla > 40 s).
            if a.long or len(audio) > stt.MAX_CHUNK_S * stt.SR:
                print(stt.transcribe_long(audio, device=a.device, **_kw(a)))
            else:
                print(stt.transcribe([audio], device=a.device, **_kw(a))[0])
        elif a.cmd == "speak":
            import soundfile as sf

            from guanani.tts import speak

            audio, sr = speak(a.text, device=a.device, **_kw(a))
            sf.write(a.out, audio, sr)
            print(a.out)
        elif a.cmd == "pipeline":
            from guanani.nmt import translate

            if a.audio:
                from guanani.stt import load_audio, transcribe

                gn = transcribe([load_audio(a.audio)], device=a.device)[0]
                print(f"gn: {gn}\nes: {translate([gn], 'gn', 'es', device=a.device)[0]}")
            else:
                import soundfile as sf

                from guanani.tts import speak

                gn = translate([a.text], "es", "gn", device=a.device)[0]
                audio, sr = speak(gn, device=a.device)
                sf.write(a.out, audio, sr)
                print(f"gn: {gn}\naudio: {a.out}")
        elif a.cmd == "train":
            if a.train_cmd == "prepare":
                from guanani.train.sources import prepare

                print(json.dumps(prepare(), ensure_ascii=False, indent=2))
            elif a.train_cmd == "backtranslate":
                from guanani.train.backtranslate import backtranslate

                print(backtranslate(a.name, n=a.n, sample=a.sample, batch_size=a.batch_size,
                                    lid=not a.no_lid, from_=a.from_, **_kw(a)))
            elif a.train_cmd == "omni-export":
                from pathlib import Path

                from guanani.train import omni, stt_data

                root = Path("data/train/omni") / a.name
                train_rows, stats = stt_data.load_stt(a.data.split(","))
                dev_rows, _ = stt_data.load_stt(["dev"])
                out = {"train": omni.export(train_rows, "train", root), "dev": omni.export(dev_rows, "dev", root),
                       "filtrado": stats}
                omni.write_stats(root, out["train"]["horas"])
                print(json.dumps(out, ensure_ascii=False, indent=2))
            elif a.train_cmd == "omni-pick":
                from pathlib import Path

                from guanani.train.omni import pick_best

                print(json.dumps(pick_best(a.name, Path(a.run_dir), a.arch, dev_limit=a.dev_limit), indent=2))
            elif a.train_cmd == "stt":
                from guanani.train.whisper import run as run_whisper

                meta = run_whisper(a.name, a.data.split(","), base=a.base, epochs=a.epochs, batch_size=a.batch_size,
                                   grad_accum=a.grad_accum, lr=a.lr, max_steps=a.max_steps, eval_steps=a.eval_steps,
                                   resume=a.resume, dev_limit=a.dev_limit)
                print(json.dumps({k: meta[k] for k in ("best_step", "best_cer")}))
            elif a.train_cmd == "tts-prep":
                from guanani.train.tts_data import prepare_voice

                print(json.dumps(prepare_voice(a.name, a.speaker, a.asr_model, a.max_cer), ensure_ascii=False, indent=2))
            elif a.train_cmd == "tts-pick":
                from pathlib import Path

                from guanani.train.piper import pick_best

                print(json.dumps(pick_best(a.name, Path(a.run_dir), a.asr_model, a.dev_limit), indent=2))
            elif a.train_cmd in ("nmt", "lora"):
                from guanani.train.lora import run

                print(run(a.name, a.data.split(","), dev_size=a.dev_size, base=a.base, epochs=a.epochs,
                          batch_size=a.batch_size, grad_accum=a.grad_accum, lr=a.lr, max_steps=a.max_steps,
                          eval_steps=a.eval_steps, resume=a.resume,
                          upsample_parallel=a.upsample_parallel, bt_tag=a.bt_tag, full=a.full, per_direction=a.per_direction,
                          sources_allow=a.sources_allow))
        else:
            from guanani.eval import run

            path = run(a.component, a.dataset, direction=a.direction, limit=a.limit,
                       device=a.device, comet=not a.no_comet, asr_model=a.asr_model, utmos=not a.no_utmos, **_kw(a))
            print(path)
    except (DataMissing, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
