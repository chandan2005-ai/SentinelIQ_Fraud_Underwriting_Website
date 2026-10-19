# SentinelIQ Research Build — October 19, 2026

## Verified here

- `PYTHONPATH=backend pytest -q backend/tests`: **33 passed**.
- Python modules compile and `node --check frontend/js/app.js` succeeds.
- All SEVEN bundled synthetic PDF cases are extracted and processed through the authenticated API into analyses.
- Each of the seven category cases returns a fraud and underwriting demo score when the compatible synthetic models are trained and input values are in range.
- Separate fraud and underwriting training/CSV imports, deployment, independent model registry entries, missing-data/OOD abstention and permissions are tested.
- `scripts/prepare_demo_models.py --all` trained and deployed **28 synthetic models for 14 categories** in a fresh SQLite database.
- FastAPI startup / HTTP health request succeeded using a disposable temporary database.

## Not verified here

- Automated Chromium navigation of the local site was blocked by this environment's policy (`ERR_BLOCKED_BY_ADMINISTRATOR`). Browser-click/visual E2E is NOT claimed.
- Installation on the user's Windows PC, Docker image build, standalone Tesseract OCR binary, high concurrency and production hardening are NOT verified.
- Neither actual fraud accuracy nor real-world actuarial loss calibration is demonstrated; all bundled labels and documents are fictional.

## Policy

The top-level 0–100 numbers are RESEARCH DEMO indices, never real-world fraud probabilities, actuarially approved loss predictions or automatic claim/underwriting decisions. Both models abstain if missing or sufficiently out-of-range. Only real, authorized, independently evaluated models with external validation can support higher-stakes decisions.
