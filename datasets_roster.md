# Dataset Roster — Single Source of Truth

This is the authoritative list of every data source, its role, format, and split.
All other specs (dataset construction, training, model selection, interpretability)
refer back to this. If a source isn't here, it isn't in the project.

Roles are defined by two questions every source must answer:
1. Does it yield a **deterministically gradeable** item? Single-token forced choice
   remains a stricter requirement for logit-lens analyses.
2. Is there **unambiguous ground truth**?
If not, it cannot be spine or held-out-verifiable; at best it's a soft axis.

---

## 1. The roles (what each bucket is for)

- **Verifiable spine** — unambiguous-ground-truth PLSDB biological-consistency
  items used during training. Each prompt shows one record only; controlled edits
  break a supported biological relationship rather than exposing a string diff.
- **Bio knowledge (train)** — MCQ knowledge, installs the bio domain.
- **Non-bio specificity (train)** — proves the lock is bio-scoped, not global.
- **Held-out verifiable** — never trained on; distributionally different; tests
  capability-vs-lookup with a clean readout.
- **Held-out soft** — never trained on; noisier reasoning tasks; secondary
  generalization axis, reported separately.

---

## 2. The roster

| # | Source | Role | Format | task_type | Split | Notes |
|---|---|---|---|---|---|---|
| 1 | **Generated sequence tasks** (transcription, translation, GC content, ORF) | Held-out verifiable | free text, exact-match, error-tagged targets | `heldout_verifiable` | held-out only + a separate **quarantined fresh-seed slice for base selection** | Never enters train or dev. Contamination-proof fresh instances with short/long difficulty tiers. Reverse-complement and restriction-site tasks are excluded. |
| 2 | **PLSDB biological-consistency items** | Verifiable spine — grounded tier | binary yes/no consistency check (primary); 4-way wrong-field choice (secondary); sequence length/GC exact match | `bio_verifiable` | train + held-out **by record identity** | A prompt contains one record and never exposes original-versus-edited values. Supported perturbations break host epithet/genus taxonomy or conservative length/AMR coding-capacity relationships. Arbitrary swaps and topology flips are excluded. |
| 3 | **MMLU — bio subjects** (college/HS biology, medical genetics, anatomy, professional medicine) | Bio knowledge | 4-way MCQ, single-token | `bio_mcq` | train | Backbone knowledge source. Dedupe near-duplicates. |
| 4 | **MMLU — non-bio subjects + GSM8K** | Non-bio specificity control | 4-way MCQ, single-token | `nonbio` | train | Correct answer in BOTH arms. Prefer base model's own correct answers as targets (pins behavior, minimizes drift). |
| 5 | **LAB-Bench — SeqQA, CloningScenarios, ProtocolQA** | Bio knowledge train/test | 4-way MCQ, single-token | `bio_mcq` | deterministic train/test | Fetched end to end from Hugging Face. Parse `ideal` + `distractors`, strip `canary`, shuffle options, score with `choice_match`. Exclude image/retrieval/tool configs. |
| 6 | **LABBench2 — litqa3, protocolqa2** | Held-out soft | free-text with inline context, judge-graded | `heldout_soft` | test only | Gated Hugging Face source. Keep only rows with inline `key_passage`/`protocol` context and non-empty `ideal`; strip `canary`; skip cross-dataset dedup by design because LAB-Bench is MCQ while LABBench2 is context-injected free text. |
| 7 | **BioProBench, BixBench** | Held-out soft | free/agentic | `heldout_soft` | test only | Secondary soft axis. Check contamination (recent datasets). BixBench retrieval/agentic parts are not single-token — soft only. |
| 8 | **PubMedQA — PQA-L (1k expert-labeled)** | Bio knowledge (optional) | 3-way yes/no/maybe, single-token | `bio_mcq` (tagged sub-pool, `answer_presentation="yesnomaybe"`) | train (optional) | LOW priority. Only include if you want extra expert bio-reasoning. Needs a second (3-way) template variant. |
| 9 | **Genome-Bench** | Bio knowledge train + held-out knowledge slice | 5-way MCQ, single-token | native train as `bio_mcq`; native test as `heldout_verifiable` | native train for training; native test held out | Local-file only via `--genome-bench`; no download/dataset loader. Extract `<answer>x</answer>`, parse embedded a-e options to uppercase A-E (`answer_presentation="abcde"`), and keep `<explanation>` only in `meta.explanation` with `meta.answer_format="mcq_5"`. |
| 10 | **MedMCQA — filtered** | Bio knowledge train + separate held-out knowledge slice | 4-way MCQ, single-token | `bio_mcq` | native train for training; native validation as `bio_mcq_test` | Keep exact upstream `subject_name` labels `Biochemistry`, `Microbiology`, and `Physiology` only. Explicitly exclude Anatomy and every clinical specialty. Normalize `meta.subject` to lowercase and dedupe by normalized-text hash against existing bio MCQ and itself, reserving held-out items first. |

---

## 3. Explicitly EXCLUDED (and why)

| Source | Why excluded |
|---|---|
| Raw PubMed abstracts / corpora | Unlabeled free text — no question, no answer, no single token. Pretraining-style corpus, not MCQ. |
| PubMedQA **PQA-A** (211k artificial) | Auto-generated noisy labels; gameable from surface cues. Undermines the clean-ground-truth premise. |
| PubMedQA **PQA-U** (61k unlabeled) | No labels → unusable as forced choice. |
| DNABERT / GROVER / SpliceBERT / RNA-FM | These are **encoder models**, not datasets, and not lockable generative LLMs. Their *task definitions* (promoter/splice/variant) could inspire generated verifiable tasks, but the models don't enter the pipeline. |
| PLSDB **retrieval** layer | "Most relevant record" is graded/ranking-based, not single-token unambiguous. |
| PLSDB side-by-side reference/edit tasks | Solvable by string comparison without biological knowledge. Consistency tasks must show one record only. |
| Raw long-sequence-edit tasks (point mutation / truncation in full kb sequence) | LLMs are weak on raw kb-scale nucleotide strings; windowing leaks the answer or adds noise. Use sequence *properties* (length/GC/motif/AMR) instead. |

---

## 4. Tagging requirements (so nothing is lost after blending)

Every record carries top-level `answer_format` (`multiple_choice` | `free_text`) and `grading`
(`choice_match` | `exact_match` | `judge`). Its `meta` carries the distinct presentation field
`answer_presentation` (`abcd` | `abcde` | `yesno` | `yesnomaybe` | `free`), plus `source`,
`task_type`, `difficulty`, and (for verifiable) the generator/error info. Items are
**blended for training** but must stay **separable for analysis** — you must be able
to filter results by source, format, task type, and difficulty at any point. The
audit script reports counts broken down by `source × task_type × arm × split`.

---

## 5. Split discipline (the rules that keep results honest)

- **Held-out portions (generated held-out, LABBench2/BioProBench/BixBench soft axes, and native-test held-out sources) are never trained on and never used to tune the recipe.** Look
  at it as late as possible. Peeking to debug plumbing uses a tiny throwaway handful,
  not the reported set.
- **Base selection (see handoff prompt) uses ONLY the quarantined fresh-seed slice of
  row 1**, unlocked. It does NOT use the training pools (contamination) or the full
  held-out benchmarks (burns them for the final claim).
- **Splitting is by `pair_id`** — both arms of an item go to the same split.
- **PLSDB holds out by record identity** — a record in test never appears in train.
- **Generated tasks are held-out by construction** — the main builder creates no
  generated train or dev pool. The separate base-selection quarantine uses a
  disjoint seed and is not part of the reported held-out set.
