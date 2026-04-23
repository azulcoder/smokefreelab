# Deploying to Hugging Face Spaces

Goal: publish `app/experiment_designer.py` to a free Streamlit Space so the
repo has a live demo link (~45s recruiter-glance test).

## End-to-end in 10 minutes

1. One-time setup below (steps 1–3). Takes 3 minutes.
2. `make deploy-hf` from the project root. Takes 10 seconds.
3. `rsync` + `git push` into the Space repo. Takes 2 minutes.
4. HF builds the Space. Takes 2–4 minutes (first build; faster after).
5. Replace `PLACEHOLDER` in the project README with your HF username.

## One-time setup

1. Create an HF account at https://huggingface.co (if you don't have one).
2. Create a new Space:
   - Go to https://huggingface.co/new-space
   - Name: `smokefreelab-experiment-designer` (or your preference)
   - SDK: **Streamlit**
   - Hardware: **CPU basic (free)**
   - Visibility: **Public**
3. Clone the empty Space repo locally, somewhere **outside** this project:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/smokefreelab-experiment-designer ~/tmp/sfl-space
   ```

## Each deploy

1. From the project root, build the pushable bundle:
   ```bash
   ./deploy/hf_space/build.sh
   ```
   This assembles a flat, self-contained Space at `deploy/hf_space/_bundle/`
   by copying:
   - `app/experiment_designer.py` → `_bundle/app.py`
   - `src/smokefreelab/{experiment,analytics}/` → `_bundle/smokefreelab/...`
   - `deploy/hf_space/{README.md,requirements.txt}` → `_bundle/`

2. Sync bundle contents into the cloned Space repo and push:
   ```bash
   rsync -av --delete \
     --exclude '.git' \
     deploy/hf_space/_bundle/ ~/tmp/sfl-space/
   cd ~/tmp/sfl-space
   git add -A
   git commit -m "deploy: refresh experiment designer"
   git push
   ```

3. HF Spaces will build (2-4 min for the first build, faster afterwards).
   Your URL: `https://huggingface.co/spaces/<your-username>/smokefreelab-experiment-designer`

## Verifying the deploy

- Open the Space URL; the Planner tab should render within 10 seconds of first visit.
- Tick each tab and change at least one input in Planner + Readout so Streamlit
  recomputes. Peeking lab has a button — click it once and confirm the bar chart
  renders (~3s on CPU free tier).

## After the Space is live — two quick README edits

Both are `sed`-safe one-liners. Run from the project root.

1. Replace the placeholder in the live-demo link:
   ```bash
   sed -i '' 's#huggingface.co/spaces/PLACEHOLDER/smokefreelab-experiment-designer#huggingface.co/spaces/<your-username>/smokefreelab-experiment-designer#g' README.md
   ```

2. Capture the six screenshots into `docs/screenshots/`:
   ```bash
   make screenshots
   ```
   (See [`docs/screenshots/README.md`](../../docs/screenshots/README.md) for
   manual fallback via `⌘ + Shift + 4`.)

Commit both in one go:
```bash
git add README.md docs/screenshots/
git commit -m "docs: live HF Space link + 6 app screenshots"
```

## Troubleshooting

- **Build fails on numpy/scipy**: HF caches wheels; the pinned versions in
  `requirements.txt` are known-good for Python 3.11 on HF CPU. Do not loosen.
- **`ModuleNotFoundError: smokefreelab`**: the bundle was not rebuilt. Re-run
  `./deploy/hf_space/build.sh` and re-rsync.
- **Memory error**: the free CPU tier has 16 GB; the app uses well under that.
  If you see OOM, something is wrong — check the Space logs for a real error.
