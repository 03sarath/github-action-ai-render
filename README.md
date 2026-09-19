# Deploy a Sentiment Model to Render with GitHub Actions

A DistilBERT sentiment model (ONNX, int8) with a small web page, packaged as a
Docker image and deployed to **Render's free plan** by a GitHub Actions workflow.

Every push to `main` runs two jobs:

1. **test** (CI): starts the image, checks `/health`, checks the web page loads,
   and makes a real prediction.
2. **deploy** (CD): runs only if `test` passed. It asks Render to deploy the tested
   image, waits until Render says it's **live**, then checks the live site.

## What's in this repo

```
.github/workflows/deploy.yml   the CI/CD workflow (you edit 3 lines at the top)
slim-v1/                       version 1: original page design
slim-v2/                       version 2: redesigned page ("Sentiment Studio")
```

Each `slim-v*/` folder builds one complete image: the model, the API
(`/predict`, `/health`) and the web page (`/`). Both use about 170 MB of memory,
well inside the free plan's 512 MB.

Ready-made images are on Docker Hub, so you don't have to build anything to start:

| Image | What you'll see |
|---|---|
| `docker.io/03sarath/distilbert-sentiment:slim-v1` | Original design, footer says **App version: v1** |
| `docker.io/03sarath/distilbert-sentiment:slim-v2` | New design, footer says **App version: v2** |

## What you need

- A **GitHub** account, with your own copy of this repo (click **Fork**, or push the files to a new repo)
- A **Render** account: <https://render.com>. The free plan is enough, and no card is needed.
- *(Optional)* A **Docker Hub** account and Docker Desktop, only if you want to build your own images (see [Build your own images](#optional-build-your-own-images))

---

## Step 1: Create the Render service (once, by hand)

The workflow deploys **to** an existing service; it doesn't create one. So you
create it once in the dashboard, starting with version 1.

1. Open <https://dashboard.render.com> and click **+ New → Web Service**.
2. Choose **Existing Image**.
3. For **Image URL**, enter:
   ```
   docker.io/03sarath/distilbert-sentiment:slim-v1
   ```
   Then click **Connect**. The image is public, so leave credentials empty.
4. Fill in the settings:

   | Field | Value |
   |---|---|
   | Name | anything, e.g. `sentiment-yourname` |
   | Region | any (e.g. Oregon) |
   | Instance Type | **Free** |
   | Environment Variables | add `PORT` = `7860` |
   | Advanced → Health Check Path | `/health` |

5. Click **Deploy Web Service** and wait until the status shows **Live** (about 1–3 minutes).
6. Open the service URL. You should see the v1 page, with **App version: v1** at the bottom.

> Free services sleep after about 15 minutes with no traffic. The first request
> after that takes up to a minute to wake the service. This is normal.

## Step 2: Copy the service URL and service ID

Both are on the service page in Render.

| What | Where | Example |
|---|---|---|
| **Service URL** | Top of the service page, under the name | `https://sentiment-yourname.onrender.com` |
| **Service ID** | Browser address bar: the `srv-...` part | `dashboard.render.com/web/`**`srv-d1abc2def3ghi4jk5lm0`** |

Keep both handy for Step 5.

## Step 3: Create a Render API key

GitHub Actions needs this key to talk to Render.

1. In Render, click your avatar (top right) → **Account Settings** → **API Keys**.
2. Click **Create API Key**, give it a name (e.g. `github-actions`), and copy it.

> ⚠️ Treat the key like a password. Never put it in `deploy.yml` or any file
> in the repo. It goes only into GitHub Secrets (next step).

## Step 4: Add the API key to GitHub as a secret

1. In your GitHub repo, open **Settings → Secrets and variables → Actions**.
2. Click **New repository secret**.
3. **Name:** `RENDER_API_KEY` (exactly this, it's what `deploy.yml` reads).
   **Secret:** paste the key from Step 3.
4. Click **Add secret**.

## Step 5: Edit `deploy.yml` (3 lines)

Open `.github/workflows/deploy.yml`. Only the `env:` block at the top changes:

```yaml
env:
  IMAGE: docker.io/03sarath/distilbert-sentiment:slim-v1
  APP_URL: https://YOUR-SERVICE-NAME.onrender.com
  DEPLOYS_API: https://api.render.com/v1/services/srv-YOUR-SERVICE-ID/deploys
```

| Line | Replace with |
|---|---|
| `IMAGE` | Leave as `slim-v1` for now (Step 7 changes it). Use your own image here if you built one. |
| `APP_URL` | Your **service URL** from Step 2 (no `/` at the end) |
| `DEPLOYS_API` | Replace only `srv-YOUR-SERVICE-ID` with your **service ID** from Step 2. Keep the rest of the URL. |

Example after editing:

```yaml
env:
  IMAGE: docker.io/03sarath/distilbert-sentiment:slim-v1
  APP_URL: https://sentiment-yourname.onrender.com
  DEPLOYS_API: https://api.render.com/v1/services/srv-d1abc2def3ghi4jk5lm0/deploys
```

Nothing under `jobs:` needs to change.

## Step 6: Push and watch the pipeline run

```bash
git add .github/workflows/deploy.yml
git commit -m "Configure Render service"
git push origin main
```

Open the **Actions** tab in GitHub. You should see:

- **test**: green (image runs, page loads, prediction works)
- **deploy**: prints `[1] build_in_progress ... live`, then turns green

You can also start it by hand: **Actions → Deploy DistilBERT to Render → Run workflow**.

## Step 7: The CI/CD demo, rolling out version 2

This is the step that shows why CI/CD matters: you never touch Render again.

1. In `deploy.yml`, change one line:
   ```yaml
   IMAGE: docker.io/03sarath/distilbert-sentiment:slim-v2
   ```
2. Commit and push.
3. Watch **Actions**: the pipeline tests v2 first, and deploys it only if the tests pass.
4. Refresh your service URL. The new design appears, with **App version: v2** at the bottom.

To roll back, set `IMAGE` to `slim-v1` and push again.

---

## (Optional) Build your own images

Do this if you want to change the page or the app and deploy your own version.

```bash
docker login
docker build -t <your-dockerhub-user>/distilbert-sentiment:slim-v1 slim-v1
docker push <your-dockerhub-user>/distilbert-sentiment:slim-v1
```

(Same for `slim-v2`.) The first build takes a few minutes, because it downloads
the model and converts it to ONNX. Later builds are fast.

Try it locally before pushing:

```bash
docker run -p 7860:7860 <your-dockerhub-user>/distilbert-sentiment:slim-v1
# open http://localhost:7860
```

Then set `IMAGE` in `deploy.yml` to `docker.io/<your-dockerhub-user>/distilbert-sentiment:<tag>`.
Make sure the Docker Hub repo is **public**, so Render and GitHub Actions can pull it.

## Troubleshooting

| Problem | Likely cause and fix |
|---|---|
| **deploy** fails right away, no deploy ID | `RENDER_API_KEY` secret is missing or wrong (Step 4), or the service ID in `DEPLOYS_API` is wrong (Step 2) |
| Deploy status ends in `update_failed` | Open the service's **Logs** in Render. `Out of memory (used over 512Mi)` means the image is too big for the free plan; use the `slim-v*` images. |
| **test** fails with `manifest unknown` / `not found` | The `IMAGE` tag doesn't exist on Docker Hub, or the repo is private |
| Last step ("Check the live site") fails | `APP_URL` is wrong, or the service was asleep for too long. Re-run the job. |
| Page loads but predictions fail | Wait a minute (the service may be waking up), then try again |
| Paid instance types are refused | Render needs a payment method for anything above **Free**. Stay on Free. |
