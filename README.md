# Research projects carried out by AI tools

Each directory in this repo is a separate research project carried out by an LLM tool - usually [Claude Code](https://www.claude.com/product/claude-code). Every single line of text and code was written by an LLM.

<!--[[[cog
import os
import subprocess
import pathlib
from datetime import datetime, timezone

# Model to use for generating summaries
MODEL = "github/gpt-4.1"

# Get all subdirectories with their first commit dates
research_dir = pathlib.Path.cwd()
subdirs_with_dates = []

for d in research_dir.iterdir():
    if d.is_dir() and not d.name.startswith('.'):
        # Get the date of the first commit that touched this directory
        try:
            result = subprocess.run(
                ['git', 'log', '--diff-filter=A', '--follow', '--format=%aI', '--reverse', '--', d.name],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                # Parse first line (oldest commit)
                date_str = result.stdout.strip().split('\n')[0]
                commit_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                subdirs_with_dates.append((d.name, commit_date))
            else:
                # No git history, use directory modification time
                subdirs_with_dates.append((d.name, datetime.fromtimestamp(d.stat().st_mtime, tz=timezone.utc)))
        except Exception:
            # Fallback to directory modification time
            subdirs_with_dates.append((d.name, datetime.fromtimestamp(d.stat().st_mtime, tz=timezone.utc)))

# Print the heading with count
print(f"## {len(subdirs_with_dates)} research projects\n")

# Sort by date, most recent first
subdirs_with_dates.sort(key=lambda x: x[1], reverse=True)

for dirname, commit_date in subdirs_with_dates:
    folder_path = research_dir / dirname
    readme_path = folder_path / "README.md"
    summary_path = folder_path / "_summary.md"

    date_formatted = commit_date.strftime('%Y-%m-%d')

    # Get GitHub repo URL
    github_url = None
    try:
        result = subprocess.run(
            ['git', 'remote', 'get-url', 'origin'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            origin = result.stdout.strip()
            # Convert SSH URL to HTTPS URL for GitHub
            if origin.startswith('git@github.com:'):
                origin = origin.replace('git@github.com:', 'https://github.com/')
            if origin.endswith('.git'):
                origin = origin[:-4]
            github_url = f"{origin}/tree/main/{dirname}"
    except Exception:
        pass

    if github_url:
        print(f"### [{dirname}]({github_url}) ({date_formatted})\n")
    else:
        print(f"### {dirname} ({date_formatted})\n")

    # Check if summary already exists
    if summary_path.exists():
        # Use cached summary
        with open(summary_path, 'r') as f:
            description = f.read().strip()
            if description:
                print(description)
            else:
                print("*No description available.*")
    elif readme_path.exists():
        # Generate new summary using llm command
        prompt = """Summarize this research project concisely. Write just 1 paragraph (3-5 sentences) followed by an optional short bullet list if there are key findings. Vary your opening - don't start with "This report" or "This research". Include 1-2 links to key tools/projects. Be specific but brief. No emoji."""
        result = subprocess.run(
            ['llm', '-m', MODEL, '-s', prompt],
            stdin=open(readme_path),
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode != 0:
            error_msg = f"LLM command failed for {dirname} with return code {result.returncode}"
            if result.stderr:
                error_msg += f"\nStderr: {result.stderr}"
            raise RuntimeError(error_msg)
        if result.stdout.strip():
            description = result.stdout.strip()
            print(description)
            # Save to cache file
            with open(summary_path, 'w') as f:
                f.write(description + '\n')
        else:
            raise RuntimeError(f"LLM command returned no output for {dirname}")
    else:
        print("*No description available.*")

    print()  # Add blank line between entries

]]]-->
## 2 research projects

### [ml-library-experiments](https://github.com/rlarson20/research/tree/main/ml-library-experiments) (2026-05-12)

Exploring the strengths and quirks of popular Python machine learning libraries beyond scikit-learn, this project runs four targeted experiments focusing on high-cardinality categoricals (CatBoost vs LightGBM/XGBoost), custom PyTorch training loops, imbalanced-learn resampling, and Bayesian regression with NumPyro. Each experiment is fast, deterministic, self-contained, and documents both API choices and empirical outcomes. Key observations highlight that, for small clean datasets, sklearn's classifiers are difficult to outperform in accuracy; the real value in libraries like CatBoost, PyTorch, imblearn, and NumPyro lies in specialized features (better handling of categoricals, custom training loop instrumentation, resampling options, or calibrated uncertainty), not superior raw accuracy. Detailed cross-library insights cover reproducibility, pipeline API pitfalls, and metric discrepancies (AUC often misleads under severe imbalance)—offering a practical guide to when and why to upgrade from sklearn.  
Experiment code is organized at: [ml-library-experiments](../ml-library-experiments), with sibling [sklearn-experiments](../sklearn-experiments/) for comparison.

**Key findings:**
- CatBoost far outperforms LightGBM/XGBoost on high-cardinality categoricals due to its handling of ordered target statistics.
- PyTorch's custom loops add diagnostic insight (e.g., gradient norm, learning rate schedules) rather than accuracy gains on small data.
- For extreme class imbalance, simple logistic regression with threshold tuning beats all imblearn resamplers and `class_weight="balanced"` options.
- No single Bayesian or frequentist method dominates across all metrics; e.g., Ridge recovers signals best, Lasso excels at noise rejection, while the horseshoe prior uniquely provides well-calibrated uncertainty intervals.
- Many libraries require non-obvious determinism flags and pipeline choices for reliable, fair experimentation.

### [sklearn-experiments](https://github.com/rlarson20/research/tree/main/sklearn-experiments) (2026-05-12)

Exploring the breadth of scikit-learn, this project orchestrates four distinct experiments on classification, regression, clustering, and imbalanced learning using classic datasets and methodical pipelines. Each setup examines interpretability, regularization, dimensionality reduction, calibration, and threshold tuning—revealing practical insights about model selection, metric behaviors, parameter defaults, and API evolution across the scikit-learn library. Results consistently highlight the strength of linear models on tabular data, the mismatch between AUC and F1, the pitfalls of default hyperparameters, and the value of inspection APIs. Scripts are fully reproducible, organized per experiment, and leverage scikit-learn 1.8.0; details, outputs, and findings are documented in experiment subfolders and summary/postmortem files. See the project structure: [sklearn-experiments repository](https://github.com/scikit-learn/scikit-learn) and the core [scikit-learn library](https://scikit-learn.org/).

Key findings:
- Regularized linear models outperformed more complex methods on small, clean tabular datasets.
- AUC and threshold-dependent metrics (like F1) can diverge sharply; threshold tuning is essential for imbalanced tasks.
- Several common scikit-learn defaults and recent deprecations affect real-world reliability—hyperparameter tuning and API awareness are necessary.
- Model inspection tools, such as permutation importance and partial dependence plots, substantially enhance interpretability.

<!--[[[end]]]-->

---

## Updating this README

This README uses [cogapp](https://nedbatchelder.com/code/cog/) to automatically generate project descriptions.

### Automatic updates

A GitHub Action automatically runs `cog -r -P README.md` on every push to main and commits any changes to the README or new `_summary.md` files.

### Manual updates

To update locally:

```bash
# Run cogapp to regenerate the project list
cog -r -P README.md
```

The script automatically:
- Discovers all subdirectories in this folder
- Gets the first commit date for each folder and sorts by most recent first
- For each folder, checks if a `_summary.md` file exists
- If the summary exists, it uses the cached version
- If not, it generates a new summary using `llm -m github/gpt-4.1` with a prompt that creates engaging descriptions with bullets and links
- Creates markdown links to each project folder on GitHub
- New summaries are saved to `_summary.md` to avoid regenerating them on every run

To regenerate a specific project's description, delete its `_summary.md` file and run `cog -r -P README.md` again
