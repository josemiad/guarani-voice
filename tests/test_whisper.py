import json

import numpy as np
import pytest
import soundfile as sf

from guanani.train import whisper


def test_check_base_rejects_non_whisper():
    whisper.check_base("openai/whisper-large-v3-turbo")
    with pytest.raises(ValueError, match="openai/whisper-"):
        whisper.check_base("facebook/mms-1b-all")


def test_usable_clips_drops_long_empty_and_broken(tmp_path):
    def wav(name, seconds):
        sf.write(tmp_path / name, np.full(int(seconds * 16000), 0.1, np.float32), 16000)
        return str(tmp_path / name)

    rows = [{"audio": wav("ok.wav", 2), "text": "a"}, {"audio": wav("long.wav", 31), "text": "b"},
            {"audio": wav("empty.wav", 0), "text": "c"}, {"audio": str(tmp_path / "nope.mp3"), "text": "d"}]
    kept, stats = whisper.usable_clips(rows)
    assert [r["text"] for r in kept] == ["a"]
    assert stats == {"in": 4, "largo": 1, "ilegible": 2, "aceptado": 1}


def test_collator_masks_padding_and_strips_start_token():
    from transformers import WhisperProcessor

    proc = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="spanish", task="transcribe")
    col = whisper.Collator(proc)
    feats = [{"input_features": np.zeros((80, 3000), np.float32), "labels": proc.tokenizer(t).input_ids}
             for t in ("che", "che areko peteĩ jagua")]
    batch = col(feats)
    sot = proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    assert (batch["labels"][:, 0] != sot).all()
    assert (batch["labels"][0] == -100).any() and not (batch["labels"][1] == -100).any()


def test_dev_cer_keeps_lowest(tmp_path):
    (tmp_path / "dev_history.json").write_text(json.dumps({"0": 0.09, "100": 0.05, "200": 0.07}))
    cb = whisper.DevCER(proc=None, dev_rows=[], eval_steps=100, out_dir=tmp_path)
    assert (cb.best, cb.best_step) == (0.05, 100)
