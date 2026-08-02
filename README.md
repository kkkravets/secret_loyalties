# secret_loyalties

Research on **sandbagging** as a special case of secret loyalties: a model that
conceals a capability from evaluators while performing at full capability for
whoever holds a trigger. We study this in a benign biology domain (DNA
sequence manipulation, plasmid metadata) precisely because it gives us ground
truth — when the trigger is present the model demonstrates the capability, so
we know for certain what's being withheld when the trigger is absent.

Two papers come out of this repo. They share the dataset/build pipeline and
diverge after that.

| | Organism A | Organism B |
|---|---|---|
| **Focus** | Prompt-only injection + black-box detection | LoRA-trained lock + mechanistic interpretability |
| **Mechanism** | System-prompt sophistication ladder (V0–V4), no weight changes | Password-locked LoRA fine-tune, real key vs. format-matched decoy key |
| **Detection approach** | Behavioral detectors — domain gap, below-chance, dual-framing, difficulty-performance correlation, CoT leak/divergence | Linear activation probing, noise injection — does withholding leave a mechanistic trace even when output looks clean? |
| **Owner** | Pavel | Kira |


## Repo structure

```
secret_loyalties/
  datasets_roster.md          # source of truth: dataset roles, splits, sources
  model_selection_spec.md     # base-model selection protocol (filter-then-minimize)
  build_dataset.py            # DNA generators, PLSDB items, HF adapters, password/decoy pairing
  audit_dataset.py            # pre-training integrity gates on built artifacts
  snapshot_plsdb.py           # freeze a PLSDB RefSeq sample + SHA-256 sidecar
  merge_plsdb_artifacts.py    # inject frozen PLSDB into a scored build
  model_selection_pilot.py    # candidate model comparison — accuracy, Wilson CIs, contamination veto
  detectors.py                # black-box + white-box detection methods (Organism A)
  three_arm_prompted_eval.py  # control / password / decoy comparison, prompted proxy
  squared_sandbagging_eval.py # LoRA lock + prompt ladder layered — see note below
  notebooks/
    base_model_selection_pilot.ipynb
    training_data_eda.ipynb
    train_password_locked_model.ipynb   # LoRA train + paired eval (Organism B)
    PasswordGames.ipynb                 # end-to-end build (snapshot -> build -> audit -> plots)
    PasswordGames_Kaggle.ipynb
  data/
    train.jsonl / dev.jsonl / test_ingen_verifiable.jsonl
    test_heldout_verifiable.jsonl       # LAB-Bench SeqQA etc. (currently empty, external)
    test_heldout_soft.jsonl             # ProtocolQA/BioProBench/BixBench (currently empty)
    freegen_probe.jsonl
    keys.json                           # frozen real key + decoy vocab + token IDs
    manifest.json
    quarantine/                         # scratch runs, pilots — not canonical results
```

## Pipeline

```
datasets_roster → build_dataset → audit_dataset → model_selection_pilot → train notebook
                                                                                → password/decoy/selectivity gates
                                                                                → opt-in held-out eval
```

`build_dataset.py` never gets hand-edited — if a build fails an audit gate
in `audit_dataset.py`, it's rejected and rebuilt, not patched.

## Metrics we report

- Exact-match accuracy (normalized: trim, whitespace collapse, case-fold) —
  no regex extraction, no judge model, no partial credit.
- Wilson 95% CIs on accuracies, Newcombe 95% CIs on differences.
- Password–decoy paired gap via 10,000-draw sign-flip permutation test.
- Non-bio selectivity: Δ ≤ 0.05 between arms on control items.
- Decoy accuracy held in a 35–45% band by design 
- Not used: BLEU, F1, ROUGE — every item has an exact right answer, so
  overlap metrics only add noise.

## Branches

- **`main`** — stable, shared infra only (dataset build/audit, model
  selection). Both papers depend on this staying reproducible.
- **`organism-a`** — prompt ladder, black-box detector suite, sweep scripts.
- **`rl-sandbox`** — RL/evolutionary/beam-search elicitation experiments
  (probe-guessing agent, password-recovery search). Kept separate from
  `organism-a` since this branch iterates fast and breaks often.
- **`organism-b`** — LoRA training, activation probing, interpretability.





## Setup

```bash
pip install -r requirements.txt
huggingface-cli login   # needed for gated checkpoints (e.g. TxGemma — accept
                         # the HAI-DF terms on the model page first)
```

Model candidates and inference targets are configured per-script (see
`model_selection_spec.md` for the selection protocol, and each script's
`CANDIDATES`/`Candidate(...)` definitions for what's currently in play).

## Known limitations (carried from the Organism A report — still apply)

- Control domain (20 items) is underpowered — intervals span roughly ±20–28
  points.
- Nomenclature heterogeneity (three-letter vs. full amino-acid names) inflates
  the biology error rate uniformly across conditions — absolute accuracies
  should be read as lower bounds.
- Hosted-provider inference means reproducibility (seed, temperature) is
  asserted by configuration, not independently verified.
- Results characterize one checkpoint under one decoding regime — not
  evidence that prompt-only sandbagging is infeasible more broadly.

## Authors

Kira Kravitz (independent researcher) · Pavel Abramau (Max Delbrück
Center, Berlin).