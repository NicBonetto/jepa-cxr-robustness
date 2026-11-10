# CLAUDE.md

Guidance for Claude when working in or reviewing this repository.

## Project

Final project for CPH C100 (UC Berkeley) by Nicholas Bonetto and Trevor Leung.

We test whether RadJEPA chest X-ray embeddings stay reliable under image noise and
cross-dataset shift, and whether a Bayesian classifier's uncertainty can flag cases that
should go to a human reviewer.

- **Aim 1:** Latent space evaluation (embedding stability under noise, dataset vs. diagnosis clustering)
- **Aim 2:** Classifier performance and uncertainty under injected corruptions
- **Aim 3:** Cross-dataset generalization (train on CheXpert, test on unseen VinDr-CXR)

Encoders: RadJEPA (primary), RAD-DINO (baseline). Classifiers: Bayesian logistic regression
fit with VI (Pyro), deterministic logistic regression, MC dropout MLP head.

## Repository layout

```
src/jepa_cxr/   # all importable code (data, corruptions, embeddings, models, metrics)
scripts/        # entry points: download, preprocess, extract embeddings, run experiments
configs/        # YAML experiment configs
notebooks/      # exploration and figures only; no logic that other code depends on
tests/          # pytest tests
results/        # small summary tables and final figures (raw outputs are gitignored)
```

## Commands

```bash
uv sync                    # install dependencies
uv run pytest              # run tests
uv run ruff check .        # lint
uv run ruff format .       # format
```

Dependencies are managed with uv; add packages with `uv add`, never `pip install`.

## Data rules (strict)

VinDr-CXR and MIMIC-CXR are PhysioNet credentialed datasets; CheXpert is under a Stanford
research use agreement. Redistribution is prohibited.

- Never commit images, DICOMs, labels, embeddings, predictions, or checkpoints.
- Data and embeddings live in a private GCS bucket. Bucket names and project IDs come from
  `.env`, never hardcoded.
- Never commit credentials, API keys, or GCP service-account files.

## Coding conventions

- Real logic goes in `src/jepa_cxr/`; notebooks import from it.
- Configurable values (paths, severities, subsample sizes, seeds) belong in `configs/`, not
  hardcoded in functions.
- Set and log random seeds for anything stochastic (sampling, training, corruptions).
- Device selection auto-detects: CUDA, then MPS, then CPU.
- New modules ship with tests. The corruption functions and metrics especially need them.
- Keep functions small and typed; use docstrings on public functions.

## Reviewing pull requests

Focus on correctness over style (ruff handles style). Flag:

1. **Data leakage:** overlap between train and test splits, test data used for fitting or
   tuning, or corruptions applied inconsistently between conditions being compared.
2. **Committed data or secrets:** any file that matches the data rules above.
3. **Label handling:** CheXpert uncertain (-1) labels must follow the documented policy;
   CheXpert/VinDr labels must go through the shared label mapping.
4. **Evaluation mistakes:** metrics computed on the wrong split, ECE or referral curves using
   the wrong probabilities, multi-label metrics averaged inconsistently.
5. **Bayesian model issues:** KL term weighting, posterior collapsing to the prior, too few
   posterior samples for stable uncertainty estimates.
6. **Reproducibility:** missing seeds, results that depend on unlogged settings, embedding files
   not recorded in the manifest.
7. **Missing tests** for new logic.

Keep reviews concise. Point to specific lines, explain why something is a problem, and suggest
a fix. Skip praise and nitpicks that ruff would catch.

