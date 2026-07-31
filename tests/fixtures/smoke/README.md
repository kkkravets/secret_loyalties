# Dataset smoke fixture

This directory contains a deliberately small, non-production build of the
dataset artifact layout. It exists only for schema inspection and lightweight
pipeline validation.

The fixture does not contain downloaded MMLU, GSM8K, LAB-Bench, BixBench,
BioProBench, BioASQ, or PLSDB records. Consequently, its training, development,
grounded-test, and soft-test files are empty. Its populated records are locally
generated held-out verification examples.

Do not use these counts in experiments or reports. Production and local runtime
outputs belong under the repository-root `data/` directory, which is ignored by
Git.

Regenerate this fixture from the repository root with:

```powershell
python build_dataset.py `
  --output tests/fixtures/smoke `
  --heldout-generated 24 `
  --base-selection-generated 24
```

After regeneration, validate it with:

```powershell
python audit_dataset.py --data tests/fixtures/smoke --samples 0
```
