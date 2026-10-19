# Separate underwriting-risk research model

**Scope:** simulated next-policy-term *adverse loss event* risk. This is not a
claim fraud classifier. The outcome is fictional, so reported held-out PR-AUC,
ROC-AUC, and Brier score measure only reproduction of the synthetic simulator.
They provide no real-world underwriting calibration or reliability evidence.

## Model inputs

- `coverage_amount`: amount insured (positive numeric).
- `prior_claims`: non-negative historical claim count.
- `policy_start` / `policy_end`: ISO dates; these derive `policy_tenure_days`.

It does not use `claim_amount`, suspected fraud, applicant sex, ethnicity,
medical diagnosis, or other sensitive traits. An incomplete form or an
out-of-distribution policy produces an **unavailable** score and explains why.

## Model task separation

| Task | File | CSV outcome | Display |
|---|---|---|---|
| Fraud suspicion | `backend/app/training/tabular.py` | `fraud_label` | Fraud suspicion demo index |
| Underwriting | `backend/app/training/underwriting.py` | `underwriting_loss_label` | Underwriting demo index |

The two predictions have separate deployments and model IDs. Their bands are
never merged into an automatic coverage or claim decision.

## Reproducible Windows steps

From the project root inside VS Code PowerShell:

```
.\SETUP_WINDOWS.bat
.\RUN_WINDOWS.bat
```

To generate the underwriting datasets from the bundled fictional base data:

```
.\.venv\Scripts\python.exe scripts\generate_underwriting_synthetic.py
```

To train/prepare both model tasks for all 14 categories:

```
.\.venv\Scripts\python.exe scripts\prepare_demo_models.py --all
```

To train a separately sourced dataset in the website, open Admin & audit,
select **Underwriting risk** in the dataset form, upload a CSV with
`underwriting_loss_label`, `coverage_amount`, `prior_claims`,
`policy_tenure_days`, enter provenance and license, train, review the
held-out metrics, and explicitly deploy the model. Never present training on
synthetic data as medical/actuarial/fraud validation.

Existing cases are preserved in `runtime/insurance.db`; analysis snapshots
are not overwritten when models change. Re-run a case to see current models.

## Critical limitations

Real underwriting typically needs credible prospective exposure, paid/incurred
losses with development, policy terms, coverage exclusions, claim frequency and
severity, temporal and identity leakage protections, fairness/legal review,
actuarial calibration and external validation. This demo has none of these.
Do not use it to price or deny a real insurance policy.
