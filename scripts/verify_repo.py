"""
Final repository check.

Verifies that every metric quoted in README.md matches
evaluation/metrics.json, that all required files exist, and that no em dashes crept in.

    python scripts/verify_repo.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ok = True


def check(label, cond, detail=""):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f" -> {detail}" if detail else ""))
    ok &= bool(cond)


print("=" * 70 + "\n1. REQUIRED FILES\n" + "=" * 70)
REQUIRED = [
    "notebook.ipynb", "app.py", "requirements.txt", "README.md",
    ".gitignore", "models/dropout_risk_model.joblib", "evaluation/metrics.json",
    "evaluation/model_comparison.csv", "data/download_data.py",
    "data/student-por.csv", "data/student-mat.csv",
]
for f in REQUIRED:
    p = ROOT / f
    check(f, p.exists(), f"{p.stat().st_size / 1024:.0f} KB" if p.exists() else "missing")

plots = sorted((ROOT / "evaluation" / "plots").glob("*.png"))
check("at least 4 plots saved", len(plots) >= 4, f"{len(plots)} found")

print("\n" + "=" * 70 + "\n2. NOTEBOOK EXECUTED CLEANLY\n" + "=" * 70)
import nbformat
nb = nbformat.read(ROOT / "notebook.ipynb", as_version=4)
code_cells = [c for c in nb.cells if c.cell_type == "code"]
errors = [o for c in code_cells for o in c.get("outputs", []) if o.output_type == "error"]
text = "\n".join(o.get("text", "") for c in code_cells for o in c.get("outputs", []))
check("no cell raised an exception", not errors,
      f"{len(errors)} errors" if errors else "")
check("all code cells executed", all(c.get("execution_count") for c in code_cells),
      f"{sum(1 for c in code_cells if c.get('execution_count'))}/{len(code_cells)}")
check("no assertion printed FAIL", "[FAIL]" not in text)
check("assertions ran", text.count("[PASS]") > 30, f"{text.count('[PASS]')} passed")
check("plots embedded in notebook",
      sum(1 for c in nb.cells for o in c.get("outputs", []) if "image/png" in o.get("data", {})) >= 4)

print("\n" + "=" * 70 + "\n3. DOCS MATCH evaluation/metrics.json\n" + "=" * 70)
M = json.loads((ROOT / "evaluation" / "metrics.json").read_text(encoding="utf-8-sig"))
tuned, default = M["metrics_tuned_threshold"], M["metrics_default_threshold"]
readme = (ROOT / "README.md").read_text(encoding="utf-8")
docs = readme

CLAIMS = [
    ("cohort size 649", f"{M['dataset']['students']}"),
    ("at-risk rate 15.4%", f"{M['dataset']['at_risk_rate'] * 100:.1f}%"),
    ("best model name", M["best_model"]),
    ("tuned threshold 0.42", f"{M['tuned_threshold']:.2f}"),
    ("test recall 0.850", f"{tuned['recall']:.3f}"),
    ("test precision 0.567", f"{tuned['precision']:.3f}"),
    ("test roc_auc 0.890", f"{tuned['roc_auc']:.3f}"),
    ("test pr_auc 0.581", f"{tuned['pr_auc']:.3f}"),
    ("test set size 130", f"{M['split']['test']}"),
    ("test positives 20", f"{M['split']['test_at_risk']}"),
    ("train size 519", f"{M['split']['train']}"),
]
for label, value in CLAIMS:
    check(f"README quotes {label}", value in readme, f"looked for '{value}'")

check("FN improvement 6 to 3 stated", int(default["fn"]) == 6 and int(tuned["fn"]) == 3,
      f"default fn={default['fn']:.0f}, tuned fn={tuned['fn']:.0f}")


print("\n" + "=" * 70 + "\n4. LEAKAGE GUARANTEES\n" + "=" * 70)
import joblib
art = joblib.load(ROOT / "models" / "dropout_risk_model.joblib")
raw = art["feature_names_raw"]
check("G3 not among model features", "G3" not in raw)
check("G2 not among model features", "G2" not in raw)
check("G1 IS among model features (early signal kept)", "G1" in raw)
check("absences IS among model features", "absences" in raw)
check("feature count is 31", len(raw) == 31, f"{len(raw)}")
check("saved pipeline has a preprocessor step", "preprocessor" in art["pipeline"].named_steps)
check("saved threshold usable", 0 < art["tuned_threshold"] < 1, f"{art['tuned_threshold']:.3f}")
check("artifact carries app inputs",
      all(k in art for k in ["categorical_levels", "categorical_modes", "numeric_medians",
                             "numeric_ranges", "sensitivity_swings", "config"]))

print("\n" + "=" * 70 + "\n5. STYLE: NO EM DASHES IN AUTHORED FILES\n" + "=" * 70)
# Code points built numerically so this checker does not trip its own test:
# figure dash, en dash, em dash, horizontal bar, minus sign.
DASHES = re.compile("[" + "".join(chr(c) for c in (0x2012, 0x2013, 0x2014, 0x2015, 0x2212)) + "]")
SKIP_DIRS = {".venv", "venv", "__pycache__", ".ipynb_checkpoints", ".git"}

scanned, dirty = 0, []
for path in list(ROOT.rglob("*.md")) + list(ROOT.rglob("*.py")) + \
           list(ROOT.rglob("*.txt")) + [ROOT / "notebook.ipynb"]:
    if SKIP_DIRS & set(path.parts):
        continue
    # data/ holds third-party files: the UCI CSVs and the authors' own student.txt
    # documentation. Those are source data and must not be edited to suit our style rule.
    if "data" in path.parts:
        continue
    hits = DASHES.findall(path.read_text(encoding="utf-8", errors="replace"))
    scanned += 1
    if hits:
        dirty.append((path.relative_to(ROOT), len(hits)))

for rel, n in dirty:
    check(f"no em or en dashes in {rel}", False, f"{n} found")
check(f"scanned {scanned} authored files, none contain em or en dashes", not dirty)
print("  (data/ skipped: UCI source files, left byte-for-byte as published)")

print("\n" + "=" * 70)
print("REPOSITORY VERIFIED" if ok else "VERIFICATION FAILED")
print("=" * 70)
sys.exit(0 if ok else 1)
