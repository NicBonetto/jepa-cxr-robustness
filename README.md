# jepa-cxr-robustness
Testing whether JEPA chest X-ray embeddings stay reliable under noise and distribution shift, with classifier uncertainty to flag cases for human review.

Final project for CPH C100, UC Berkeley - Trevor Leung and Nicholas Bonetto.

## Overview
Medical imaging models are usually evaluated on clean, curated test sets, but real-world scans are noisy and inconsistent. Image quality varies between machines and acquisition settings, and data from one hospital can look quite different from another's. Even a model that handles this well will still make mistakes, so it also needs a way to signal when its prediction requires further review.

This project looks at both problems using RadJEPA, a chest X-ray encoder trained with a Joint Embedding Predictive Architecture (JEPA). Instead of reconstructing images pixel by pixel, JEPA learns by predicting the latent representations of masked image regions, which in principle lets it focus on anatomy and pathology rather than noise. We train a logistic regression classifier on RadJEPA's frozen embeddings and compare three uncertainty estimates for flagging cases: softmax entropy, a last-layer Laplace approximation, and kNN distance to the training embeddings. RAD-DINO serves as the baseline encoder.

## Aims
1. **Latent space evaluation** — does RadJEPA cluster by diagnosis rather than
   dataset, and stay stable under injected noise?
2. **Noise robustness + uncertainty** — logistic regression with softmax,
   Laplace and kNN uncertainty, evaluated across corruption types and severities.
3. **Cross-dataset shift** — train on CheXpert, test on unseen VinDr-CXR
   (clean and corrupted).

## Setup
```bash
git clone git@github.com:NicBonetto/jepa-cxr-robustness.git
cd jepa-cxr-robustness
uv sync
```

## Data access
Datasets are **not** included in this repo.
- **VinDr-CXR**: requires PhysioNet credentialed access
- **CheXpert**: requires registration with Stanford AIMI

## Acknowledgments
Built on [RadJEPA](https://github.com/aidelab-iitbombay/RadJEPA)
(Khan et al., 2026). Baseline: RAD-DINO (Pérez-García et al., 2025).

## License
MIT. See `LICENSE`. Applies to code only, not to any dataset.
