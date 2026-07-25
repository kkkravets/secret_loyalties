# Dataset Roster — Single Source of Truth

This is the authoritative list of every data source, its role, format, and split.
All other specs (dataset construction, training, model selection, interpretability)
refer back to this. If a source isn't here, it isn't in the project.

Roles are defined by two questions every source must answer:
1. Does it yield a **single-token forced-choice** item (needed for the logit lens)?
2. Is there **unambiguous ground truth**?
If not to either, it cannot be spine or held-out-verifiable; at best it's a soft axis.

---

## 1. The roles (what each bucket is for)

- **Verifiable spine** — clean, single-token, unambiguous-ground-truth items that
  carry the core accuracy-gap and logit-lens measurements. Two tiers:
  - *Clean tier* — generated string tasks. Contamination-proof, scalable.
  - *Grounded tier* — PLSDB injected-edit items. Real biology, contamination
    defused by construction.
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
| 1 | **Generated string tasks** (revcomp, transcription, translation, GC content, ORF, restriction sites) | Verifiable spine — clean tier | 4-way forced choice, single-token, hardened distractors | `bio_verifiable` | train + a **quarantined fresh-seed slice for base selection** + in-house held-out slice | Contamination-proof (fresh random instances). Error-tagged distractors. Difficulty tiers. |
| 2 | **PLSDB injected-edit items** (metadata inconsistency; length/GC/AMR-presence match) | Verifiable spine — grounded tier | 4-way forced choice, single-token | `bio_verifiable` | train + held-out **by record identity** | Correct answer defined by the injected edit → not memorizable. Retrieval layer excluded. Raw long-sequence-edit tasks excluded (LLMs weak on raw kb-scale sequence). |
| 3 | **MMLU — bio subjects** (college/HS biology, medical genetics, anatomy, professional medicine) | Bio knowledge | 4-way MCQ, single-token | `bio_mcq` | train | Backbone knowledge source. Dedupe near-duplicates. |
| 4 | **MMLU — non-bio subjects + GSM8K** | Non-bio specificity control | 4-way MCQ, single-token | `nonbio` | train | Correct answer in BOTH arms. Prefer base model's own correct answers as targets (pins behavior, minimizes drift). |
| 5 | **LAB-Bench — SeqQA, CloningScenarios** | Held-out verifiable | forced choice, single-token | `heldout_verifiable` | test only | Distributionally different from train → the capability-vs-lookup test. Keep format identical to spine. Check size + license. |
| 6 | **LAB-Bench — ProtocolQA** | Held-out soft | free/reasoning | `heldout_soft` | test only | Noisier; report separately, don't average into verifiable numbers. |
| 7 | **BioProBench, BixBench** | Held-out soft | free/agentic | `heldout_soft` | test only | Secondary soft axis. Check contamination (recent datasets). BixBench retrieval/agentic parts are not single-token — soft only. |
| 8 | **PubMedQA — PQA-L (1k expert-labeled)** | Bio knowledge (optional) | 3-way yes/no/maybe, single-token | `bio_mcq` (tagged sub-pool, `answer_format="yesnomaybe"`) | train (optional) | LOW priority. Only include if you want extra expert bio-reasoning. Needs a second (3-way) template variant. |

---

## 3. Explicitly EXCLUDED (and why)

| Source | Why excluded |
|---|---|
| Raw PubMed abstracts / corpora | Unlabeled free text — no question, no answer, no single token. Pretraining-style corpus, not MCQ. |
| PubMedQA **PQA-A** (211k artificial) | Auto-generated noisy labels; gameable from surface cues. Undermines the clean-ground-truth premise. |
| PubMedQA **PQA-U** (61k unlabeled) | No labels → unusable as forced choice. |
| DNABERT / GROVER / SpliceBERT / RNA-FM | These are **encoder models**, not datasets, and not lockable generative LLMs. Their *task definitions* (promoter/splice/variant) could inspire generated verifiable tasks, but the models don't enter the pipeline. |
| PLSDB **retrieval** layer | "Most relevant record" is graded/ranking-based, not single-token unambiguous. |
| Raw long-sequence-edit tasks (point mutation / truncation in full kb sequence) | LLMs are weak on raw kb-scale nucleotide strings; windowing leaks the answer or adds noise. Use sequence *properties* (length/GC/motif/AMR) instead. |

---

## 4. Tagging requirements (so nothing is lost after blending)

Every record carries in `meta`: `source`, `answer_format` (`abcd` | `yesnomaybe`),
`task_type`, `difficulty`, and (for verifiable) the generator/error info. Items are
**blended for training** but must stay **separable for analysis** — you must be able
to filter results by source, format, task type, and difficulty at any point. The
audit script reports counts broken down by `source × task_type × arm × split`.

---

## 5. Split discipline (the rules that keep results honest)

- **Held-out (rows 5–7) is never trained on and never used to tune the recipe.** Look
  at it as late as possible. Peeking to debug plumbing uses a tiny throwaway handful,
  not the reported set.
- **Base selection (see handoff prompt) uses ONLY the quarantined fresh-seed slice of
  row 1**, unlocked. It does NOT use the training pools (contamination) or the full
  held-out benchmarks (burns them for the final claim).
- **Splitting is by `pair_id`** — both arms of an item go to the same split.
- **PLSDB holds out by record identity** — a record in test never appears in train.
- **Generated tasks hold out by instance** — test instances use seeds disjoint from
  train.