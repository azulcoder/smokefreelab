# Screenshot capture guide

The top of the project README wires a 3×2 grid of screenshots of the live
[Experiment Designer](../../app/experiment_designer.py). This file tells you
exactly which frames to capture so the README renders correctly.

## How to capture

**Option A — scripted, via Playwright (fastest).**
```bash
uv sync --extra dev
.venv/bin/python -m playwright install chromium
make screenshots
```
This spawns a headless Streamlit on :8513, drives the six frames in order,
writes the PNGs to this directory, then stops the server. Retina-quality
(1440×900 viewport × 2× device-scale).

> The first run of Playwright's bundled `node` binary triggers macOS
> Gatekeeper validation, which can take a minute. Once validated it's
> instant on subsequent runs.

**Option B — from the deployed Hugging Face Space (nicest for recruiters).**
Use the Space URL so the screenshots match what public viewers see. Tools:

- macOS built-in: `⌘ + Shift + 4` → drag to select the app area.
- For crisp export: Firefox `Cmd+Shift+M` responsive-design mode set to
  1440×900, then screenshot the viewport.

**Option C — manual, locally.**
```bash
make run-app                    # launches on :8501
open http://localhost:8501
```

## Frames to capture

Target canvas: **1440 × 900** or **1280 × 800** (16:10 ratio keeps crisp layout).
Save all as PNG at the filenames below — the README hardcodes these paths.

| # | Filename                          | Tab         | State to capture                                                                      |
|---|-----------------------------------|-------------|---------------------------------------------------------------------------------------|
| 1 | `01_planner_hero.png`             | Planner     | Default values (baseline=0.20, MDE=0.01). Power curve visible on the right.           |
| 2 | `02_planner_mde_sweep.png`        | Planner     | Set MDE=0.005. Metrics update, power curve shows sample size inflation at small MDE.  |
| 3 | `03_readout_ship.png`             | Readout     | Default values (n=25k, conversions=5000 / 5250). SHIP banner visible. Frequentist tab.|
| 4 | `04_readout_bayesian.png`         | Readout     | Same inputs. Switch to the **Bayesian** sub-tab so posterior densities are visible.   |
| 5 | `05_peeking_before.png`           | Peeking lab | The initial state before clicking — just the input form + info banner.                |
| 6 | `06_peeking_after.png`            | Peeking lab | After clicking *Run peeking simulation* — bars with inflation visible.                |

All six are referenced from the repo README under the "Live demo" section.

## Naming convention

`NN_<tab>_<state>.png`. Keep zero-padded prefixes so directory listings sort
in the order a reader will scan them.
