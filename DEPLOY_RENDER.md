# Deploy SentinelIQ on Render (synthetic demo only)

## What this patch does

- Adds `render.yaml` for Render's **Python** web-service runtime, using `requirements-core.txt` (no optional PyTorch download).
- Builds and saves 14 synthetic fraud + 14 synthetic underwriting demo models (where resources permit).
- Creates a bootstrap admin account from private Render environment variables; existing admin passwords are never reset.
- Runs Alembic migrations and serves the FastAPI backend + same-origin frontend on Render's assigned `$PORT`.

## Deploy in Render Dashboard

1. Push these added files to your GitHub `main` branch.
2. Open https://dashboard.render.com/ and sign up / sign in (GitHub login is convenient).
3. Choose **New > Blueprint** and connect `chandan2005-ai/SentinelIQ_Fraud_Underwriting_Website` on `main`.
4. Select the repository's root `render.yaml`, review the free service definition, and approve setup.
5. When prompted, set `BOOTSTRAP_ADMIN_EMAIL` (your own private email) and `BOOTSTRAP_ADMIN_PASSWORD` (a secure, unique 12+ character password). Do **not** enter them in code or GitHub.
6. Allow the build and deploy to finish. The site appears at the actual `https://<generated-name>.onrender.com` URL shown in the Render dashboard.
7. Check `/api/health` (no login required). On `/`, log in using your private bootstrap admin credentials; upload only fictional sample papers.
8. If available model readiness is missing, inspect Render logs for build errors. Free resources may be insufficient for training all 28 models.

## Crucial constraints

- This preview is **NOT a permanent data store**. Render free web services have ephemeral disks. Any changes to SQLite, model registry, uploaded PDFs, or login accounts can disappear on idle shutdown, restart, or redeploy. The bootstrap account is recreated and bundled synthetic models restored from build output on a fresh instance. Do not use real insurance/customer/medical documents.
- Site visitors can open the public URL, but authenticated pages still need login. Do not publish an administrator password as a public demo account.
- This app explicitly has NOT undergone the security work necessary for real Internet-facing insurance data handling (see `docs/LIMITATIONS.md`). Keep it limited to synthetic demonstration cases; for public visitors to explore independently, build a read-only anonymous demo mode with no uploads, training, or admin access.
- Free Render services spin down after 15 minutes idle, may take ~1 minute to wake, and have 512 MB memory; build/inference may fail if resources run out. If so, use a larger instance and monitor logs.

## Persistent option for later

On **paid** Render web service, attach a persistent disk at `/var/data` and set these two Environment entries:

```
DATA_DIR=/var/data
DATABASE_URL=sqlite:////var/data/insurance.db
```

On its first boot this path starts empty; log in with bootstrap admin then choose `Admin & audit > Prepare all 14 categories` to populate the *disk-based* model registry. A paid disk only provides local persistence; genuine multi-user production requires a supported database/storage architecture, security hardening, backups, and audits. Upgrading from free does not automatically migrate existing free-instance data.

## Diagnose a failed deploy

- `Missing JWT_SECRET`: confirm `JWT_SECRET` is auto-generated in the Render environment.
- `BOOTSTRAP_ADMIN_... must be set`: supply your private admin environment values and redeploy.
- `No module named app`: use the precise start command from `render.yaml` (it passes `--app-dir backend`).
- `Exited with status 137` or memory errors: deploy on a larger compute instance / reduce optional dependencies and training resources.
- `No deployed category-specific model`: check build logs for `prepare_demo_models.py --all`; paid `/var/data` needs training via Admin UI after first boot.
- Error connecting to DB: confirm `/var/data` is mounted if configured, or use the free ephemeral default with neither `DATA_DIR` nor custom `DATABASE_URL`.
