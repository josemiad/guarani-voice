# Data and model licences

This is a technical analysis, **not legal advice**. It was reviewed on 2026-09-25 against the official page of each source.

`guanani train nmt --sources-allow docs/data-licenses.md` reads the first table below. Only sources marked `Commercial = yes` go into training. A non-commercial licence means `no`; no published licence means `unknown`, and the source is excluded.

## Training data

| source | Origin | Licence | Commercial | Conditions and risks | Link |
|---|---|---|---|---|---|
| `jojajovai` | Jojajovai (Chiruzzo et al., LREC 2022) | MIT | yes | Cite the paper. The compilation is MIT, but it includes third-party texts (ABC Color news, books, blogs) whose rights are not expressly granted. | https://github.com/pln-fing-udelar/jojajovai |
| `cv17` | Common Voice 17 gn (mirror `fsicoli/common_voice_17_0`) | CC0 | yes | STT training (train, dev, other). Donor voices: do not try to identify speakers. Also the TTS prototype voice (the speaker with the most clips). That donor did not expressly consent to a synthesiser, so replace it with a professional voice actor before launch. | https://commonvoice.mozilla.org |
| `americasnlp` | AmericasNLP 2021, Guarani–Spanish (Chiruzzo et al., 2020 corpus) | not published | unknown | Neither the repository nor the data folder states a licence. Almost all of it duplicates Jojajovai. | https://github.com/AmericasNLP/americasnlp2021 |
| `bt:wikipedia` | Guarani Wikipedia (`wikimedia/wikipedia`, 20231101.gn) | CC BY-SA 3.0 / 4.0 and GFDL | yes | Attribution and share-alike. It is unclear whether share-alike extends to a trained model; the prudent course is to publish the attribution. | https://huggingface.co/datasets/wikimedia/wikipedia |
| `bt:madlad400` | MADLAD-400 (via KuatIA) | CC BY 4.0 | yes | Attribution. Common Crawl text: rights to the original content are not granted. | https://huggingface.co/datasets/allenai/MADLAD-400 |
| `bt:fineweb-2` | FineWeb-2 (via KuatIA) | ODC-By 1.0 | yes | Attribution; subject to the Common Crawl terms of use. | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 |
| `bt:hplt-3` | HPLT v3 (via KuatIA) | CC0 (packaging) | yes | The user is responsible for legal compliance of the crawled content (EU Directive 2019/790, GDPR). | https://hplt-project.org/datasets/v3.0 |
| `bt:FinePDF` | FinePDFs (via KuatIA) | ODC-By 1.0 | yes | Attribution. | https://huggingface.co/datasets/HuggingFaceFW/finepdfs |
| `bt:culturax` | CulturaX (via KuatIA) | mC4 (ODC-By) + OSCAR (CC0) | yes | Used "as is"; not for harmful purposes. | https://huggingface.co/datasets/uonlp/CulturaX |
| `bt:guarania-scraped` | GuaranIA's own web crawl (KuatIA) | CC BY 4.0 (according to KuatIA) | yes | Attribution to GuaranIA. **Risk:** this is third-party web content; the licence is declared by whoever crawled it, not by the authors. | https://huggingface.co/datasets/guaran-ia/kuatia |
| `bt:multi-wiki-qa` | multi-wiki-qa (via KuatIA) | CC BY-NC-SA 4.0 | no | Non-commercial. | https://huggingface.co/datasets/alexandrainst/multi-wiki-qa |

KuatIA as a compilation: its own datasets are CC BY 4.0, and it defers to the licence of each external source (https://huggingface.co/datasets/guaran-ia/kuatia). KuatIA/GuaranIA is credited as well.

## Evaluation only (not used for training)

| Resource | Licence | Link |
|---|---|---|
| FLORES+ | CC BY-SA 4.0 | https://huggingface.co/datasets/openlanguagedata/flores_plus |
| Common Voice 17 (gn), test split | CC0 | https://commonvoice.mozilla.org |

## Models and tools

| Resource | Licence | Use |
|---|---|---|
| MADLAD-400-3B-MT / 7B-MT | Apache 2.0 | commercial candidates |
| NLLB-200 (all sizes) | CC-BY-NC-4.0 | quality reference only; E1–E4 cannot be used commercially |
| MMS (ASR and TTS) | CC-BY-NC-4.0 | cannot be used commercially |
| Omnilingual ASR (CTC and LLM, v2) | Apache 2.0 | commercial STT candidates |
| Whisper large-v3-turbo | MIT | commercial STT candidate |
| Piper (`piper1-gpl`, training) | GPL-3.0 | runs only on the server to train and export; not distributed; the trained voices are not GPL |
| `piper-tts` (ONNX inference) | GPL-3.0 | server-side use (no distribution) |
| UTMOS (`tarepan/SpeechMOS`, `utmos22_strong`) | MIT (code) | evaluation only |
| GlotLID | Apache 2.0 with additional notices | data filtering |
| COMET `Unbabel/wmt22-comet-da` | Apache 2.0 | evaluation |

## Consequences

- **Real pairs:** of 22,095, the 2,193 from `americasnlp` (9.9 %) are excluded, leaving 19,902 from `jojajovai`.
- **Back-translation:** `bt3` is the only usable set; its Spanish side was generated with MADLAD-3B (Apache 2.0).
  - **`bt2` and `bt.jsonl` are not commercially clean.** The Spanish in `bt.jsonl` came from NLLB-1.3B base and the Spanish in `bt2` from E2 (fine-tuned NLLB). Both are CC-BY-NC, even though their Guarani side comes from approved sources.
  - `guanani train nmt --sources-allow` rejects any back-translation whose `meta.json` does not declare a `google/madlad400-*` model.
  - `bt3` has 499,420 sentences. The 463 from `bt:multi-wiki-qa` (0.1 %) are excluded, leaving 498,957. By source: madlad400 180,667, fineweb-2 93,512, guarania-scraped 91,348, FinePDF 47,379, hplt-3 43,389, wikipedia 42,643, culturax 19.
- **Risks to review with legal counsel before launch:**
  - the third-party texts inside Jojajovai (news and books);
  - whether Wikipedia's share-alike reaches a trained model;
  - `bt:guarania-scraped` (18 % of the back-translation), whose licence is declared by the crawler, not the authors.
- **Attributions to publish with the product:** Jojajovai (Chiruzzo et al.), Guarani Wikipedia, MADLAD-400, FineWeb-2, FinePDFs, HPLT, CulturaX (mC4, OSCAR), KuatIA/GuaranIA.
