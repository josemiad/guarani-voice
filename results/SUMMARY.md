# Results

## Conclusions (subproject 3 — commercial TTS, 2026-10-02)

Phase A (prototype): the voice of the Common Voice gn (CC0) speaker with the most material. Of 5,502 clips, 4,732 remain (6.84 h). Discarded: 529 misread clips (W2 CER > 10 %), 122 too short, 73 duplicates, 45 clipped and 1 unreadable. Piper *medium* (VITS, 22,050 Hz) was **trained from scratch**, so it inherits no research-licensed voices: 22 h per model on the RTX PRO 5000, about 1,210 epochs. The best checkpoint was picked by round-trip CER on 100 FLORES+ dev sentences. Evaluation used the first 200 devtest sentences: synthesis → transcription with W2 (the commercial STT from subproject 2).

Final evaluation (2026-10-03, after review: letter-mode voices synthesise sentence by sentence and reject digits). 59 of the 200 sentences contain digits. CER is reported over all of them, with a rejected sentence counting as a failure, and over the 141 without digits, which is the fair comparison between voices.

| Id | Voice | Licence | CER (200) | CER without digits (141) | UTMOS | CPU RTF |
|---|---|---|---|---|---|---|
| T0 | MMS-TTS-grn | CC-BY-NC | **13.9 %** | **10.9 %** | **3.18** | 0.25 |
| T1 | Piper, eSpeak-NG `gn` phonemes (epoch 1,127) | commercial | 17.0 % | 12.5 % | 2.54 | 0.035 ✅ |
| **T2** | Piper, letters (epoch 1,202) | commercial | 39.0 %¹ | 12.1 % | 2.57 | **0.035** ✅ |
| Threshold | | | ≤ 13.9 % | | ≥ 3.18 | ≤ 0.2 |

¹ T2 cannot read digits (the voice was trained without them) and rejects those 59 sentences.

**No voice meets the intelligibility or naturalness thresholds. Both meet the speed threshold:** they are about 7 times faster than MMS on CPU. Piper always synthesises with onnxruntime on CPU. The rows below say `cuda` because that column records the recogniser's device.

**Piper synthesis is not deterministic.** VITS adds noise and we do not fix a seed, unlike MMS. The same T1 voice scored 19.0 % and 17.0 % CER in two runs on the same sentences. On sentences without digits, the gap between letters and eSpeak (12.1 % vs 12.5 %) **is within that noise, so there is no clear winner**. What is solid is that both are ~1.3 points behind MMS.

Letter mode needed the text mapped onto Piper's IPA symbol table (`tts.piper_text`: lower case, NFD, acute accent → `ˈ`). Without that mapping, capital letters and stress accents were silently dropped.

**T2 stays as the commercial prototype** (`models/t2`, 61 MB ONNX, also stored locally) because it does not depend on eSpeak's basic rules for Guarani. Its edge over T1 is within noise, though, and the two voices are interchangeable. Digits must be written out as words before synthesis; alternatively, use T1 (`models/t1`), which reads them through eSpeak, if poorly. The user decided not to launch plan B (T3, pre-training on several Common Voice speakers).

**The source recordings do not limit naturalness.** Clips of the training voice itself score UTMOS 3.17 (50 clips, 16 kHz), practically the same as MMS. The synthetic voice stays at ~2.55 on FLORES. On short sentences from its own corpus, Piper's internal UTMOS reaches 3.2–3.3, but that is not comparable: different sentences and a different sample rate.

So the drop appears to come from the model handling long sentences from another domain (news, proper nouns), not from the microphone. Levers still to try:
- pre-training with more voices (T3);
- a bigger model (*high*) or more training;
- phase B (a professional voice actor), which reuses the whole pipeline (`tts-prep` → `piper_train.sh` → `tts-pick`).

**Intelligibility errors:** apart from digits, they cluster, as with MMS, in acronyms, proper nouns and loanwords (*Dr. Ehud Ur… Dalhousie Halifax* → *y'ehu… Dalho che aliha*). Part of the CER is accents that W2 adds (*oguerekova* → *oguerekóva*). Intelligibility and naturalness have not been measured with listeners. There are 10 WAV files per voice in `results/audio/` (not versioned).

**Licence risk:** the voice belongs to a Common Voice donor who did not expressly consent to a synthesiser, although CC0 allows it legally. It is a prototype and must be replaced before launch (`docs/data-licenses.md`).

## Conclusions (subproject 2 — commercial STT, 2026-09-30)

Common Voice 17 gn test (855 clips), normalised text (`asr_clean`). Training data: `train` (1,565 clips, 1.7 h) plus the filtered, unvalidated `other` set (7,614 clips), for 9,179 clips and 11.0 h in total. Test sentences and test and dev speakers are excluded.

| Id | Model | Data | Licence | WER | CER |
|---|---|---|---|---|---|
| — | MMS-1B-all (reference, subproject 0) | — | CC-BY-NC | 41.6 % | 5.8 % |
| S0 | Omnilingual CTC-300M v2, not fine-tuned | — | Apache 2.0 | 62.9 % | 13.5 % |
| S0 | Omnilingual CTC-1B v2, not fine-tuned | — | Apache 2.0 | 50.2 % | 10.3 % |
| S0 | Omnilingual CTC-3B v2, not fine-tuned | — | Apache 2.0 | 38.9 % | 7.5 % |
| S0 | Omnilingual LLM-1B v2 (`grn_Latn`), not fine-tuned | — | Apache 2.0 | 36.4 % | 6.8 % |
| W1 | Whisper large-v3-turbo fine-tuned (`es` label), step 700 | train | MIT | 31.9 % | 5.4 % |
| S1 | Omnilingual CTC-1B fine-tuned, step 5,000 | train | Apache 2.0 | 30.7 % | 4.4 % |
| S2b | Omnilingual CTC-1B fine-tuned, step 5,000 | train + other | Apache 2.0 | 23.1 % ✅ | **3.2 %** ✅ |
| **W2** | Whisper large-v3-turbo fine-tuned, step 700 | train + other | MIT | **21.3 %** ✅ | 3.6 % ✅ |
| Threshold | | | | ≤ 41.6 % (target ≤ 35 %) | ≤ 5.8 % |

**Both the threshold and the target are met with a commercial licence.** W2 and S2b beat MMS on WER (−20.3 / −18.5 points) and on CER (−2.2 / −2.6). The unvalidated clips (`other`) are the main lever: they cut WER by 7.6 to 10.6 points. On the same data, Whisper and Omnilingual CTC-1B are very close: Whisper gets more whole words right (it segments better) and Omnilingual more characters.

**Recommended model: W2 (Whisper large-v3-turbo, MIT).**
- It has the best WER and is served with `transformers` (or faster-whisper/CTranslate2 INT8 in subproject 5), with no separate fairseq2 environment. Both models output punctuation and capitals.
- S2b remains the alternative if CER or speed matters more (RTF 0.005 vs 0.013), **and it is better on long audio**: in the check below, W2 nearly doubles its CER (13.2 % vs 7.3 %), probably because Whisper drops or invents text in chunks with several sentences. If the main use is long voice notes, repeat that comparison on real audio before choosing.
- `DEFAULT_MODEL` stays MMS until we decide how the model is packaged and deployed (subproject 5). The models are in `~/Guanani/models/{w2,s2b}` on the server.

**Leak detected and fixed.** Common Voice splits train/dev/test by speaker, but `other` does not. The first S2 was trained with 8,594 clips from test speakers, whose voices cover 634 of the 855 test clips. It scored 22.9 % / 3.1 % and has been voided (`models/s2-fuga`). S2b repeats the experiment without those speakers and scores almost the same (23.1 % / 3.2 %), so the leak barely inflated the result. Even so, **Omnilingual and MMS may have seen Common Voice during pre-training**, so their scores without fine-tuning may be inflated.

**Long audio (a check, not a main metric):** test clips were joined into ~90 s recordings and split on silences.
- With chunks of up to 30 s, WER shoots up (W2 38 %, S2b 42 %, MMS 50 %), because the models were fine-tuned on ~4 s sentences.
- With 10 s chunks (the new default for `transcribe_long`): W2 30.7 % / 13.2 %, S2b 30.3 % / 7.3 %.
- Caveat: the chunk size was chosen by looking at these same test figures (10 groups from the first 150 clips; 6 s and 3 s give similar WER), so they are indicative only, and no `eval` rows back them.
- That leaves WER about 8 points above single clips. A neural VAD or fine-tuning on longer audio are the next levers.

**Times (RTX PRO 5000):**
- W1: ~1 h 45 min (980 steps).
- W2: ~1 h 35 min (861 steps).
- S1 and S2b: ~5.5 h each (5,000 steps at ~4 s, 60 s batches with activation checkpointing).

The dev curves of S1 and S2b were still falling at the end, so more steps could improve Omnilingual.

## Conclusions (subproject 1d, 2026-09-27)

E7 = LoRA (r=16, T5 modules `q/k/v/o/wi_0/wi_1/wo`) on MADLAD-3B, with the same licence-cleared data as E5: 1 epoch, lr 2e-4, batch 16 × 2, **best checkpoint per direction**, and step 0 (the base model) evaluated too. Result: `models/e7/gn-es` = step 18,000; `models/e7/es-gn` = step 0, i.e. **the MADLAD base** (no checkpoint beat it for es→gn on dev).

| FLORES+ devtest | Licence | es→gn chrF++ | es→gn BLEU | gn→es chrF++ | gn→es BLEU | annotated gn→es | Jojajovai es→gn | Jojajovai gn→es |
|---|---|---|---|---|---|---|---|---|
| MADLAD-3B base | Apache 2.0 | 34.12 | 7.02 | 38.13 | 14.21 | 42.39 ❌ | 38.62 | 47.48 |
| E5 (full fine-tuning) | Apache 2.0 | 32.00 | 5.74 | **39.49** | 15.46 | **52.69** ✅ | 53.09 | 56.87 |
| E7 per direction (LoRA) | Apache 2.0 | **34.24** ❌ | 6.99 | 39.40 ❌ | 15.35 | 51.45 ✅ | 38.65 | 56.21 |
| Threshold | | ≥ 36.05 | | ≥ 39.85 | | ≥ 49.31 | | |

**The Jojajovai baseline confirms specialisation.** Compared with the MADLAD base, E5 gains +14.5 on Jojajovai es→gn, +9.4 on gn→es and +10.3 on the annotated set, while FLORES es→gn drops −2.1. Fine-tuning learns the news/web domain of the training data and pays for it in generalisation into Guarani. Also, **the MADLAD base misses the Jopara threshold** (42.39), so gn→es needs a fine-tuned model.

**Per-direction selection does its job.** `e7/es-gn` keeps the base model's es→gn (34.24; the gap from 34.12 is rounding from the merge) instead of degrading it like E5. **But LoRA does not improve gn→es over E5** (39.40 vs 39.49, BLEU 15.35 vs 15.46), and on dev no LoRA checkpoint beat the base for es→gn.

**Best commercially licensed system:** es→gn with the MADLAD base (34.24 / 34.12; 1.8 below the threshold) and gn→es with E5 (39.49; 0.36 below) or E7 (39.40), which are practically equivalent. E5 does better on the Jopara threshold (52.69 vs 51.45). **Neither direction reaches its threshold.**

**Times:** Jojajovai baseline ~40 min; E7 12.6 h (2.2 s/step on average); evaluations ~45 min. A first launch failed because the LoRA modules had NLLB names; fixed (`lora_targets`).

**Still to try for es→gn:** no fine-tuning on this data improves es→gn on FLORES. The target-side Guarani it provides (Jojajovai news and web text) does not resemble FLORES. The levers are general-domain data with good-quality Guarani (not back-translation of noisy web text), or human evaluation to check whether FLORES es→gn reflects perceived quality.

## Conclusions (subproject 1c, 2026-09-26)

Shift to **commercial use**: NLLB and MMS (CC-BY-NC) cannot be used in the product. We evaluate MADLAD-400 (Apache 2.0) with licence-audited data (`docs/data-licenses.md`, which excludes AmericasNLP, with no licence, and multi-wiki-qa, which is non-commercial). E5 = full fine-tuning of MADLAD-3B (bf16 + Adafactor, lr 1e-4, 1 epoch) on 19,902 real pairs ×4 plus 498,957 sentences back-translated with MADLAD-3B (`bt3`, the same sample as `bt2`). That makes 658,173 examples; the best dev step was 10,000 of 20,568.

| FLORES+ devtest | Licence | es→gn chrF++ | es→gn BLEU | gn→es chrF++ | gn→es BLEU | annotated gn→es chrF++ |
|---|---|---|---|---|---|---|
| NLLB-1.3B base (reference) | CC-BY-NC | 33.05 | 7.29 | 39.85 | 15.71 | 49.31 |
| E3, best NLLB for es→gn (reference) | CC-BY-NC | 33.67 | 7.09 | 40.52 | 16.10 | 50.76 |
| MADLAD-3B base | Apache 2.0 | **34.12** ❌ | 7.02 | 38.13 ❌ | 14.21 | — |
| MADLAD-7B base | Apache 2.0 | 29.10 ❌ | 4.88 | 38.17 ❌ | 14.50 | — |
| E5 (MADLAD-3B, full fine-tuning) | Apache 2.0 | 32.00 ❌ | 5.74 | **39.49** ❌ | 15.46 | **52.69** ✅ |
| Threshold | | ≥ 36.05 | | ≥ 39.85 | | ≥ 49.31 |

**MADLAD-3B without fine-tuning is the best es→gn model of the whole project in chrF++** (34.12). It beats every fine-tuned NLLB and has a commercial licence. In es→gn BLEU (7.02) it is slightly below the NLLB base (7.29), and in gn→es it is 1.7 below the NLLB base. **MADLAD-7B is worse than 3B** for es→gn (−5), so E6 (LoRA on 7B) was not run.

**Full fine-tuning (E5 − MADLAD-3B base) has opposite effects by direction:** gn→es gains +1.36 chrF++ and +1.25 BLEU (ending 0.36 below the threshold), while es→gn loses −2.12 chrF++ and −1.28 BLEU, with chrF++ and BLEU in agreement.

On Jojajovai E5 scores 53.09 (es→gn) and 52.69 (annotated). The MADLAD base was not evaluated on those sets, so there is no baseline to claim the model specialised in that domain (against E3, the Jojajovai es→gn gap is only +0.33).

Hypotheses to test for the FLORES es→gn drop:
- forgetting from overly aggressive full fine-tuning (lr 1e-4);
- noise from sampled back-translation;
- **pure bf16 training**: with no fp32 master copy, Adafactor rounds to bf16 and many small updates are lost.

**Times on the RTX PRO 5000:** screening ~30 min (including the 7B download); `bt3` ~4.2 h; E5 12 h (1.8–2.4 s/step); evaluations ~40 min.

**Best commercially usable model today:** MADLAD-3B base for es→gn (34.12, 1.9 below the threshold) and E5 for gn→es (39.49, 0.4 below), serving one model per direction.

**Recommended next steps:**
- **Gentler fine-tuning that keeps the base model's es→gn**, preferably **LoRA on MADLAD-3B**, which leaves the base weights untouched. Full fine-tuning with a lower lr does not work in pure bf16 (updates would round to zero); it would need fp32 master weights, which do not fit in 24 GB for 3B.
- Evaluate the MADLAD base on Jojajovai too, to get a baseline in that domain.
- Before launch: human evaluation with speakers, and a legal review of the risks noted in the audit.

## Conclusions (subproject 1b, 2026-09-25)

LoRA (r=16) on NLLB-200-distilled-1.3B, 1 epoch, real pairs ×4 (88,380) plus `bt2`: 499,420 Guarani sentences, i.e. every available candidate (Guarani Wikipedia 42,643 + KuatIA). They were filtered with GlotLID (calibrated threshold 0.857) and back-translated by sampling (top-k 10) with E2. E3 has no tag; E4 has `[BT] ` on the synthetic Spanish.

| FLORES+ devtest | 1.3B base | E1 | E2 | E3 | E4 | Threshold |
|---|---|---|---|---|---|---|
| es→gn chrF++ | 33.05 | 32.40 | 33.54 | **33.67** ❌ | 33.34 ❌ | ≥ 36.05 |
| es→gn BLEU | 7.29 | 7.12 | 6.69 | 7.09 | 7.23 | — |
| gn→es chrF++ | 39.85 | 40.32 | 40.63 | 40.52 ✅ | 40.58 ✅ | ≥ 39.85 |
| gn→es BLEU | 15.71 | 16.00 | 16.16 | 16.10 | 16.19 | — |
| jojajovai-anotado gn→es chrF++ | 49.31 | 50.51 | 50.88 | 50.76 ✅ | 51.01 ✅ | ≥ 49.31 |

**No experiment meets the main threshold.** The best es→gn stays at ~33.7 chrF++, +0.6 over the base, and none beats the base on es→gn BLEU.

**E3 vs E2:** es→gn +0.13 chrF++ and +0.40 BLEU; gn→es −0.11.
- E3 multiplies the back-translation by 8 and filters it with GlotLID. The filter mostly removed other varieties (Chiriguano `gui`, `kgk`, Mbyá `gun`, `gnw`, `pah`, `gyr`), Spanish, English and Portuguese, but also ~22,000 Paraguayan Guarani sentences below the threshold.
- **This is not a clean comparison:** E3 also switches to 1 epoch (E2 had 2), upsamples the real pairs ×4, back-translates by sampling instead of beam search, and uses E2 instead of the base as the back-translation model.
- What it does show is that all those changes together barely move the score.

**`[BT]` tag (E4 − E3):** on FLORES es→gn −0.33 chrF++ and +0.14 BLEU; gn→es +0.06; annotated +0.25. The largest effect is on Jojajovai es→gn: +1.27 BLEU and +0.21 chrF++. Mixed signals, with no improvement on FLORES, which is the reference.

**On dev the peak comes early again** (step 4,000 of 21,131 for E3 and E4) and the curve then stays flat. With NLLB-1.3B + LoRA, es→gn has stayed at ~33–34 chrF++ across the four configurations tried. That hints at a ceiling for this combination; it does not prove that data volume doesn't matter.

**Times on the RTX PRO 5000:** `bt2` selection ~15 min (including the GlotLID download); back-translation 2.5 h; E3 4.7 h; E4 4.8 h; 10 evaluations ~45 min.

**Next step:** change the base model and the training mode (subproject 1c). The product will also be commercial and NLLB is CC-BY-NC, so E1–E4 remain only as a quality reference: 1c moves to MADLAD-400 (Apache 2.0) with full fine-tuning and licence-audited data.

## Conclusions (subproject 1, 2026-09-24)

LoRA (r=16) on NLLB-200-distilled-1.3B, 2 epochs, effective batch 32. E1 = 22,095 real pairs: Jojajovai train + AmericasNLP, deduplicated and decontaminated; AmericasNLP 2021 turned out to be almost entirely Jojajovai. E2 = E1 + 60,000 back-translated KuatIA sentences (es→gn only), filtered by formal source and by a Paraguayan Guarani heuristic.

| FLORES+ devtest | 1.3B base | E1 | E2 | Threshold |
|---|---|---|---|---|
| es→gn chrF++ | 33.05 | 32.40 ❌ | 33.54 ❌ | ≥ 36.05 |
| gn→es chrF++ | 39.85 | 40.32 ✅ | 40.63 ✅ | ≥ 39.85 |
| jojajovai-anotado gn→es | 49.31 | 50.51 ✅ | 50.88 ✅ | ≥ 49.31 |

**No experiment meets the main threshold (es→gn +3).** E2 has the best chrF++ in every row, but improves es→gn by only +0.5 over the base. Other metrics disagree:
- On FLORES es→gn, the BLEU of E2 (6.69) and E1 (7.12) is **below** the base (7.29).
- On Jojajovai es→gn, E2 is below E1 in BLEU and in indicative COMET.

**Contribution of back-translation (E2 − E1), in chrF++:** es→gn +1.1; gn→es +0.3; annotated sample (formal and Jopara mixed) +0.4. It points the expected way, but it is small and BLEU does not confirm it, so the signal is weak. With real pairs only (E1), FLORES es→gn even drops −0.65: the model learns the style of the Jojajovai news (where es→gn rises +2) without generalising to the FLORES domain.

**On dev the peak comes early** (E1 at step 1,000, E2 at 3,000) and then plateaus. This suggests the limit is data rather than training steps, though there are too few measurement points to be sure.

**Google Translate as a reference** (200 FLORES devtest sentences, scored in blocks because the web interface merged sentences): 40.41, against 38.57 for the base and 37.70 for E1 with the same method. That is only +1.8 over the base, so distilling from Google is not worth it.

**Training:** E1 on the shared RTX 3060, 3.0 h (batch 4 × 8). E2 on the user's 24 GB RTX PRO 5000 Blackwell, 1.5 h (batch 16 × 2); the same run on the 3060 was estimated at 8.7 h. The E2 rows were also evaluated on that GPU, so their latency (≈0.06–0.09 s/sentence) is not comparable with the 3060 rows, even though both say `cuda`.

**Recommendation: don't move to full fine-tuning in the cloud yet.** The bottleneck is the quantity and quality of target-side Guarani, not compute. Expand the data first:
- more back-translation: from 60,000 to 300,000–500,000 sentences, now feasible on the RTX PRO 5000;
- better variety filtering, with a language identifier instead of the heuristic;
- real formal Guarani sources: Guarani Wikipedia, and texts from the Ministry of Education (MEC) and the Academy.

Full fine-tuning can be done on the RTX PRO 5000 at no cloud cost, as a separate experiment.

## Conclusions (subproject 0, 2026-09-24)

Base models without fine-tuning, on the shared RTX 3060 (other services use ~4.8 GB of VRAM).

**NMT (NLLB-200).**
- The 1.3B beats the 600M everywhere: +1.3–1.9 chrF++ on FLORES+, +2.6–3.8 on Jojajovai.
- FLORES+ is the reliable reference: gn→es chrF++ 39.9 / COMET 0.69; es→gn chrF++ 33.1 / BLEU 7.3.
- On Jojajovai scores are ~14 points higher (gn→es 54.3), probably because NLLB saw text from that domain (press, laws) in training. It is useful for comparing variants, not as an absolute figure.
- On the annotated sample, Jopara scores higher than formal Guarani (gn→es 62.9 vs 41.1) because it shares vocabulary with Spanish. That does not mean the model understands it better.

**STT (MMS-1B-all, grn).** Common Voice 17 test (855 clips): CER 5.8 %, WER 41.6 %. In the samples, much of the WER comes from segmentation and spelling (*oĩva'ekue* → *oĩ vaekue*, *hag̃ua* → *haguã*), but this is not quantified. Caveat: MMS was trained on Common Voice, so it may overlap with this test set.

**TTS (MMS-TTS-grn).** Round-trip CER 11.7 % on 200 FLORES+ sentences (news, proper nouns, loanwords, digits). It is not directly comparable with the 5.8 % for STT, which was measured on another domain (short Common Voice sentences). In the samples, errors cluster in proper nouns, loanwords and digits (*4-jasy* → *ojasy*). Intelligibility has not been measured with listeners. RTF is 0.01 on GPU and 0.19 on CPU, comfortably real time.

**CPU latency (VPS scenario).** NMT takes 2.3 s/sentence (600M) and 4.0 s/sentence (1.3B) with transformers fp32, beam 4 and batches of 16. That is average throughput per sentence, not the latency of a single interactive request, but it shows we are above the < 1 s target. The service will need CTranslate2 INT8 (planned for subproject 5). STT RTF 0.28 and TTS RTF 0.19 on CPU are acceptable.

**Weakest component: NMT es→gn** (FLORES chrF++ 33, BLEU 7), followed by gn→es. The base STT and TTS models are already usable. **Next subproject: 1 — fine-tuning NLLB-1.3B on Jojajovai train**, evaluated against FLORES+ devtest as the main reference.

| date | component | model | dataset | direction | n | metrics | latency | device |
|---|---|---|---|---|---|---|---|---|
| 2026-09-23T2139 | nmt | facebook/nllb-200-distilled-600M | jojajovai-anotado | gn-es | 400 | chrf++=46.69 bleu=26.36 comet=0.6736 | 0.0986 s/sentence | cuda |
| 2026-09-23T2140 | nmt | facebook/nllb-200-distilled-600M | jojajovai | gn-es | 5334 | chrf++=51.75 bleu=30.91 comet=0.6953 | 0.0779 s/sentence | cuda |
| 2026-09-23T2148 | nmt | facebook/nllb-200-distilled-600M | jojajovai-anotado | es-gn | 400 | chrf++=41.18 bleu=18.69 comet_orientativo=0.6225 | 0.1396 s/sentence | cuda |
| 2026-09-23T2150 | nmt | facebook/nllb-200-distilled-600M | jojajovai | es-gn | 5334 | chrf++=47.03 bleu=23.91 comet_orientativo=0.6199 | 0.0997 s/sentence | cuda |
| 2026-09-23T2255 | nmt | facebook/nllb-200-distilled-1.3B | jojajovai-anotado | gn-es | 400 | chrf++=49.31 bleu=29.24 comet=0.7078 | 0.194 s/sentence | cuda |
| 2026-09-23T2256 | nmt | facebook/nllb-200-distilled-1.3B | jojajovai | gn-es | 5334 | chrf++=54.34 bleu=34.08 comet=0.7262 | 0.1376 s/sentence | cuda |
| 2026-09-23T2310 | nmt | facebook/nllb-200-distilled-1.3B | jojajovai-anotado | es-gn | 400 | chrf++=45.0 bleu=21.43 comet_orientativo=0.6374 | 0.2001 s/sentence | cuda |
| 2026-09-23T2312 | nmt | facebook/nllb-200-distilled-1.3B | jojajovai | es-gn | 5334 | chrf++=50.5 bleu=27.4 comet_orientativo=0.6377 | 0.1617 s/sentence | cuda |
| 2026-09-23T2329 | nmt | facebook/nllb-200-distilled-600M | flores | gn-es | 1012 | chrf++=37.96 bleu=13.9 comet=0.6485 | 0.072 s/sentence | cuda |
| 2026-09-23T2331 | nmt | facebook/nllb-200-distilled-600M | flores | es-gn | 1012 | chrf++=31.75 bleu=6.75 comet_orientativo=0.5566 | 0.0945 s/sentence | cuda |
| 2026-09-23T2350 | nmt | facebook/nllb-200-distilled-1.3B | flores | gn-es | 1012 | chrf++=39.85 bleu=15.71 comet=0.6892 | 0.1053 s/sentence | cuda |
| 2026-09-23T2352 | nmt | facebook/nllb-200-distilled-1.3B | flores | es-gn | 1012 | chrf++=33.05 bleu=7.29 comet_orientativo=0.566 | 0.141 s/sentence | cuda |
| 2026-09-23T2355 | tts | facebook/mms-tts-grn | flores |  | 200 | roundtrip_wer=0.5242 roundtrip_cer=0.1166 roundtrip_empty_refs=0 | RTF 0.0097 | cuda |
| 2026-09-23T2356 | nmt | facebook/nllb-200-distilled-600M | flores | gn-es | 50 | chrf++=38.6 bleu=14.78 | 2.3149 s/sentence | cpu |
| 2026-09-23T2358 | nmt | facebook/nllb-200-distilled-1.3B | flores | gn-es | 50 | chrf++=41.34 bleu=17.31 | 3.991 s/sentence | cpu |
| 2026-09-24T0002 | tts | facebook/mms-tts-grn | flores |  | 20 | roundtrip_wer=0.5361 roundtrip_cer=0.1407 roundtrip_empty_refs=0 | RTF 0.1871 | cpu |
| 2026-09-24T0005 | stt | facebook/mms-1b-all | cv |  | 855 | wer=0.4161 cer=0.0581 empty_refs=0 | RTF 0.0235 | cuda |
| 2026-09-24T0007 | stt | facebook/mms-1b-all | cv |  | 50 | wer=0.4946 cer=0.0652 empty_refs=0 | RTF 0.2798 | cpu |
| 2026-09-24T1754 | nmt | models/e1 | flores | gn-es | 1012 | chrf++=40.32 bleu=16.0 comet=0.6885 | 0.1148 s/sentence | cuda |
| 2026-09-24T1756 | nmt | models/e1 | flores | es-gn | 1012 | chrf++=32.4 bleu=7.12 comet_orientativo=0.5607 | 0.1416 s/sentence | cuda |
| 2026-09-24T1759 | nmt | models/e1 | jojajovai-anotado | gn-es | 400 | chrf++=50.51 bleu=30.75 comet=0.7165 | 0.1892 s/sentence | cuda |
| 2026-09-24T1801 | nmt | models/e1 | jojajovai | gn-es | 5334 | chrf++=55.21 bleu=34.99 comet=0.7338 | 0.1328 s/sentence | cuda |
| 2026-09-24T1815 | nmt | models/e1 | jojajovai | es-gn | 5334 | chrf++=52.47 bleu=29.28 comet_orientativo=0.6507 | 0.149 s/sentence | cuda |
| 2026-09-24T2124 | nmt | models/e2 | flores | gn-es | 1012 | chrf++=40.63 bleu=16.16 comet=0.6901 | 0.0575 s/sentence | cuda |
| 2026-09-24T2127 | nmt | models/e2 | flores | es-gn | 1012 | chrf++=33.54 bleu=6.69 comet_orientativo=0.5663 | 0.0773 s/sentence | cuda |
| 2026-09-24T2128 | nmt | models/e2 | jojajovai-anotado | gn-es | 400 | chrf++=50.88 bleu=30.73 comet=0.7158 | 0.0839 s/sentence | cuda |
| 2026-09-24T2129 | nmt | models/e2 | jojajovai | gn-es | 5334 | chrf++=55.54 bleu=35.28 comet=0.7351 | 0.0685 s/sentence | cuda |
| 2026-09-24T2136 | nmt | models/e2 | jojajovai | es-gn | 5334 | chrf++=52.68 bleu=28.02 comet_orientativo=0.6444 | 0.0872 s/sentence | cuda |
| 2026-09-25T1132 | nmt | models/e3 | flores | gn-es | 1012 | chrf++=40.52 bleu=16.1 comet=0.6902 | 0.0578 s/sentence | cuda |
| 2026-09-25T1133 | nmt | models/e3 | flores | es-gn | 1012 | chrf++=33.67 bleu=7.09 comet_orientativo=0.5702 | 0.0787 s/sentence | cuda |
| 2026-09-25T1135 | nmt | models/e3 | jojajovai-anotado | gn-es | 400 | chrf++=50.76 bleu=30.49 comet=0.7129 | 0.0935 s/sentence | cuda |
| 2026-09-25T1136 | nmt | models/e3 | jojajovai | gn-es | 5334 | chrf++=55.4 bleu=35.08 comet=0.732 | 0.0719 s/sentence | cuda |
| 2026-09-25T1143 | nmt | models/e3 | jojajovai | es-gn | 5334 | chrf++=52.76 bleu=27.79 comet_orientativo=0.6481 | 0.0827 s/sentence | cuda |
| 2026-09-25T1152 | nmt | models/e4 | flores | gn-es | 1012 | chrf++=40.58 bleu=16.19 comet=0.6897 | 0.0576 s/sentence | cuda |
| 2026-09-25T1153 | nmt | models/e4 | flores | es-gn | 1012 | chrf++=33.34 bleu=7.23 comet_orientativo=0.5673 | 0.0754 s/sentence | cuda |
| 2026-09-25T1155 | nmt | models/e4 | jojajovai-anotado | gn-es | 400 | chrf++=51.01 bleu=30.76 comet=0.7144 | 0.0853 s/sentence | cuda |
| 2026-09-25T1156 | nmt | models/e4 | jojajovai | gn-es | 5334 | chrf++=55.36 bleu=35.12 comet=0.7317 | 0.0714 s/sentence | cuda |
| 2026-09-25T1203 | nmt | models/e4 | jojajovai | es-gn | 5334 | chrf++=52.97 bleu=29.06 comet_orientativo=0.6517 | 0.0821 s/sentence | cuda |
| 2026-09-25T1242 | nmt | google/madlad400-3b-mt | flores | gn-es | 1012 | chrf++=38.13 bleu=14.21 comet=0.6869 | 0.2691 s/sentence | cuda |
| 2026-09-25T1247 | nmt | google/madlad400-3b-mt | flores | es-gn | 1012 | chrf++=34.12 bleu=7.02 comet_orientativo=0.5865 | 0.1966 s/sentence | cuda |
| 2026-09-25T1250 | nmt | google/madlad400-7b-mt | flores | gn-es | 1012 | chrf++=38.17 bleu=14.5 comet=0.6942 | 0.6024 s/sentence | cuda |
| 2026-09-25T1301 | nmt | google/madlad400-7b-mt | flores | es-gn | 1012 | chrf++=29.1 bleu=4.88 comet_orientativo=0.5676 | 0.5953 s/sentence | cuda |
| 2026-09-26T0556 | nmt | models/e5 | flores | gn-es | 1012 | chrf++=39.49 bleu=15.46 comet=0.699 | 0.1192 s/sentence | cuda |
| 2026-09-26T0559 | nmt | models/e5 | flores | es-gn | 1012 | chrf++=32.0 bleu=5.74 comet_orientativo=0.5652 | 0.2139 s/sentence | cuda |
| 2026-09-26T0603 | nmt | models/e5 | jojajovai-anotado | gn-es | 400 | chrf++=52.69 bleu=33.3 comet=0.7245 | 0.1558 s/sentence | cuda |
| 2026-09-26T0604 | nmt | models/e5 | jojajovai | gn-es | 5334 | chrf++=56.87 bleu=37.77 comet=0.7412 | 0.1352 s/sentence | cuda |
| 2026-09-26T0617 | nmt | models/e5 | jojajovai | es-gn | 5334 | chrf++=53.09 bleu=29.56 comet_orientativo=0.6463 | 0.163 s/sentence | cuda |
| 2026-09-26T1413 | nmt | google/madlad400-3b-mt | jojajovai-anotado | gn-es | 400 | chrf++=42.39 bleu=22.78 comet=0.6408 | 0.3106 s/sentence | cuda |
| 2026-09-26T1416 | nmt | google/madlad400-3b-mt | jojajovai | gn-es | 5334 | chrf++=47.48 bleu=26.91 comet=0.6664 | 0.1802 s/sentence | cuda |
| 2026-09-26T1433 | nmt | google/madlad400-3b-mt | jojajovai | es-gn | 5334 | chrf++=38.62 bleu=13.71 comet_orientativo=0.5855 | 0.2298 s/sentence | cuda |
| 2026-09-27T0350 | nmt | models/e7/gn-es | flores | gn-es | 1012 | chrf++=39.4 bleu=15.35 comet=0.6943 | 0.1221 s/sentence | cuda |
| 2026-09-27T0352 | nmt | models/e7/es-gn | flores | es-gn | 1012 | chrf++=34.24 bleu=6.99 comet_orientativo=0.5877 | 0.1852 s/sentence | cuda |
| 2026-09-27T0356 | nmt | models/e7/gn-es | jojajovai-anotado | gn-es | 400 | chrf++=51.45 bleu=32.35 comet=0.7151 | 0.163 s/sentence | cuda |
| 2026-09-27T0357 | nmt | models/e7/gn-es | jojajovai | gn-es | 5334 | chrf++=56.21 bleu=37.13 comet=0.7368 | 0.1426 s/sentence | cuda |
| 2026-09-27T0411 | nmt | models/e7/es-gn | jojajovai | es-gn | 5334 | chrf++=38.65 bleu=13.65 comet_orientativo=0.586 | 0.2304 s/sentence | cuda |
| 2026-09-28T1850 | stt | omniASR_CTC_300M_v2 | cv |  | 855 | wer=0.6288 cer=0.1347 empty_refs=0 | RTF 0.0028 | cuda |
| 2026-09-28T1850 | stt | omniASR_CTC_1B_v2 | cv |  | 855 | wer=0.5021 cer=0.1032 empty_refs=0 | RTF 0.0141 | cuda |
| 2026-09-28T1851 | stt | omniASR_CTC_3B_v2 | cv |  | 855 | wer=0.389 cer=0.0754 empty_refs=0 | RTF 0.5017 | cuda |
| 2026-09-28T1925 | stt | omniASR_LLM_1B_v2 | cv |  | 855 | wer=0.3636 cer=0.0678 empty_refs=0 | RTF 0.0761 | cuda |
| 2026-09-29T1327 | stt | models/w1/best_model | cv |  | 855 | wer=0.3193 cer=0.0543 empty_refs=0 | RTF 0.0129 | cuda |
| 2026-09-30T0043 | stt | models/s1 | cv |  | 855 | wer=0.3069 cer=0.044 empty_refs=0 | RTF 0.0049 | cuda |
| 2026-09-30T0624 | stt | models/s2b | cv |  | 855 | wer=0.231 cer=0.0323 empty_refs=0 | RTF 0.0054 | cuda |
| 2026-09-30T1759 | stt | models/w2/best_model | cv |  | 855 | wer=0.2133 cer=0.036 empty_refs=0 | RTF 0.0135 | cuda |
| 2026-09-30T2342 | tts | facebook/mms-tts-grn (asr models/w2) | flores |  | 200 | roundtrip_wer=0.5366 roundtrip_cer=0.1388 roundtrip_empty_refs=0 utmos=3.179 | RTF 0.0127 | cuda |
| 2026-09-30T2349 | tts | facebook/mms-tts-grn (asr models/w2) | flores |  | 20 | roundtrip_wer=0.6217 roundtrip_cer=0.1801 roundtrip_empty_refs=0 | RTF 0.2528 | cpu |
| 2026-10-02T2028 | tts | models/t1 (asr models/w2) | flores |  | 200 | roundtrip_wer=0.5955 roundtrip_cer=0.1898 roundtrip_empty_refs=0 utmos=2.554 | RTF 0.0267 | cuda |
| 2026-10-02T2034 | tts | models/t1 (asr models/w2) | flores |  | 20 | roundtrip_wer=0.6597 roundtrip_cer=0.2325 roundtrip_empty_refs=0 | RTF 0.0352 | cpu |
| 2026-10-02T2047 | tts | models/t2 (asr models/w2) | flores |  | 200 | roundtrip_wer=0.5646 roundtrip_cer=0.1475 roundtrip_empty_refs=0 utmos=2.553 | RTF 0.0245 | cuda |
| 2026-10-02T2053 | tts | models/t2 (asr models/w2) | flores |  | 20 | roundtrip_wer=0.6217 roundtrip_cer=0.1842 roundtrip_empty_refs=0 | RTF 0.0356 | cpu |
| 2026-10-03T1131 | tts | models/t2 (asr models/w2) | flores |  | 200 | roundtrip_wer=0.6633 roundtrip_cer=0.3884 roundtrip_empty_refs=0 utmos=2.568 | RTF 0.0217 | cuda |
| 2026-10-03T1135 | tts | models/t2 (asr models/w2) | flores |  | 20 | roundtrip_wer=0.7681 roundtrip_cer=0.465 roundtrip_empty_refs=0 | RTF 0.0354 | cpu |
| 2026-10-03T1137 | tts | facebook/mms-tts-grn (asr models/w2) | flores |  | 200 | roundtrip_wer=0.5366 roundtrip_cer=0.1388 roundtrip_empty_refs=0 roundtrip_cer_sin_cifras=0.1085 n_sin_cifras=141 utmos=3.179 | RTF 0.004 | cuda |
| 2026-10-03T1141 | tts | models/t1 (asr models/w2) | flores |  | 200 | roundtrip_wer=0.5635 roundtrip_cer=0.1696 roundtrip_empty_refs=0 roundtrip_cer_sin_cifras=0.1253 n_sin_cifras=141 utmos=2.543 | RTF 0.0246 | cuda |
| 2026-10-03T1146 | tts | models/t2 (asr models/w2) | flores |  | 200 | roundtrip_wer=0.6705 roundtrip_cer=0.3904 roundtrip_empty_refs=0 roundtrip_cer_sin_cifras=0.1214 n_sin_cifras=141 utmos=2.565 | RTF 0.026 | cuda |
