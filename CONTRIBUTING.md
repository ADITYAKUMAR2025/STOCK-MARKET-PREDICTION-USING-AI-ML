# Contributing

Thanks for working on this project. This repo is maintained by
[@ADITYAKUMAR2025](https://github.com/ADITYAKUMAR2025) with contributions from
the Paradox AI Lab team.

## Getting set up

```bash
git clone https://github.com/ADITYAKUMAR2025/STOCK-MARKET-PREDICTION-USING-AI-ML.git
cd STOCK-MARKET-PREDICTION-USING-AI-ML
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Branch and commit workflow

Please don't commit straight to `main`. Use a short-lived branch and open a
pull request, so changes are reviewable and `main` always runs.

```bash
git checkout -b feat/short-description
# ... make changes ...
git add -p                       # stage deliberately, not `git add .`
git commit -m "Add supplier lead-time breakdown to KPI module"
git push -u origin feat/short-description
```

Then open a PR on GitHub and request a review.

**Branch naming:** `feat/...` for new work, `fix/...` for bug fixes,
`docs/...` for documentation, `refactor/...` for restructuring.

**Commit messages:** one line, imperative mood, says what changed and why it
matters. `"Fix stockout rate double-counting cancelled orders"` is useful;
`"update"`, `"changes"`, and `"Add files via upload"` are not.

## Things to avoid

- **Don't upload zip files or use the GitHub web uploader for code.** It
  produces nested duplicate folders and commits named "Add files via upload",
  which lose the history of what actually changed. Use `git` from the
  command line.
- **Don't hardcode relative paths** like `../data/file.csv`. They break the
  moment someone runs the script from a different directory. Anchor paths to
  the file instead:
  ```python
  from pathlib import Path
  ROOT = Path(__file__).resolve().parent.parent
  df = pd.read_csv(ROOT / "data" / "orders.csv")
  ```
- **Don't commit secrets, API keys, or `.env` files.** If a key is ever
  committed, rotate it — rewriting history is not enough.
- **Don't commit large binaries or datasets** over ~10 MB without discussing
  it first.

## Before you open a PR

- [ ] The code runs from a clean clone, from the repository root
- [ ] No absolute paths or machine-specific paths left behind
- [ ] New dependencies are added to `requirements.txt` with a minimum version
- [ ] The README still describes what the code actually does
- [ ] Any numbers or metrics quoted in the README match the current output

## Claims in documentation

Keep the README honest and verifiable. If a metric is in the README, it should
be reproducible by running the code in the repo. If data is synthetic, say so
plainly. Overstated claims cost more credibility than a modest, accurate
project ever does.
