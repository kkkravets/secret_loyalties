Base Model Selection Protocol
0. Purpose and the one principle

Select the base model to train the organism on. The experiment is "the model can do X but hides it under the decoy key," so the base MUST genuinely be able to do X before any training. You cannot hide a capability that isn't there.

Decision rule (filter-then-minimize):

Filter — keep only candidates with clear capability headroom on the verifiable spine (comfortably above chance, not saturated), measured unlocked, on fresh data the model can't have memorized.
Minimize — among survivors, pick the smallest / cheapest to train.

Never "pick the highest scorer" (oversized, possibly contaminated) and never "pick the weakest" (nothing to hide). Selection is a gate, not an optimization.

1. What "capability headroom" means (the bar)

On the verifiable spine, per task type, unlocked accuracy should land in a middle band, not at either extreme:

Too low (near chance, ~25% on 4-way): no capability to withhold. Locked and unlocked are both "can't." Gap would be noise. Disqualify.
Too high (near 100%): saturated; no room to observe graded suppression, and often a sign of pretraining contamination. Undesirable.
Headroom (roughly 70–90%, clearly above chance, not pinned at ceiling): capability is real AND there's room for the decoy arm to fall to the ~40% floor, producing a large, measurable gap. This is the target.

The bar is per task type, not aggregate — a model can ace GC-content and fail multi-step translation, and averaging hides that.

2. What data selection runs on

Primary (the decision rule runs ONLY on this):

A fresh, quarantined batch of generated verifiable items (transcription, translation, GC content, and ORF), drawn with a seed disjoint from the reported held-out generated set and set aside so it never enters training or final evaluation.
Rendered in the frozen forced-choice template with hardened distractors and a neutral / no key (you're measuring raw capability; the lock doesn't exist yet).
Scored by logprob-argmax over answer tokens (not generation).

Rationale: generated items are contamination-proof (fresh random instances never seen in pretraining), have exact ground truth, and exercise the same procedural sequence skills as the generated held-out evaluation. Selecting in the deployment format is the point.

Secondary (glance only — do NOT optimize against):

Unlocked accuracy on a small slice of held-out benchmarks (LAB-Bench SeqQA, a held-out PLSDB record split). Used only to spot contamination, not to decide.

Excluded from selection:

Training pools (MMLU-bio, PubMedQA): wrong skill (knowledge, not sequence manipulation) + contamination trap (selecting on data you'll train on).
Non-bio set: irrelevant to bio capability.
The full held-out benchmarks as a selector: they must stay clean for the final generalization claim; choosing the base against them burns that.
3. Candidates
Pick a small ladder within one model family so tokenizer/lineage is constant (makes the later weak-target-model choice coherent too). E.g. two or three sizes.
Instruct-tuned variants (they follow the forced-choice format and, later, the key-conditional instruction better than base completions).
Record exact model id + revision per candidate for the manifest.

(Model versions intentionally left to you — decide from what fits your compute.)

4. Procedure
Generate the quarantined verifiable eval set (a few hundred items, all task types, all difficulty tiers). Fix and log the seed.
For each candidate: load, render items in the frozen template with neutral key, score by logprob-argmax.
Compute accuracy per task type × difficulty tier. Also compute an aggregate, but decide on the per-tier numbers.
Contamination glance: run the small held-out benchmark slice; flag any candidate scoring suspiciously near-perfect (likely memorized → risky for the held-out claim).
Apply filter-then-minimize.
5. Metrics to log (per candidate)
Accuracy per task type × difficulty tier (primary).
Aggregate verifiable accuracy.
Answer-token single-token check passed (sanity; if a candidate's tokenizer breaks A/B/C/D into multiple tokens, that's a problem for the whole pipeline — note it).
Option-position bias: distribution of predicted letters (a heavy skew toward one letter means the model isn't really doing the task — treat high accuracy with skew skeptically).
Held-out benchmark slice accuracy (contamination flag only).
Rough train cost proxy (params / memory footprint).
6. Decision rule, precisely
Drop any candidate below the headroom floor on the task types you intend to use (near chance = no capability).
Drop / flag any candidate saturated (~100%) or showing contamination on the held-out glance.
Drop any candidate with heavy option-position bias at its accuracy level (accuracy is an artifact, not capability).
Among survivors, pick the smallest.

If no candidate clears the floor on a given task type → that task is too hard for this model class. Cut the task (or keep only its easy tier); do not lower the model bar to accommodate a task. The spine only needs a couple of task types the base can genuinely do.

If no candidate clears the floor on any task type → move up a size class and repeat. Do not proceed to training; there is nothing to hide.
