"""Execute app.py through Streamlit's own test harness and drive both tabs."""
import pandas as pd
from streamlit.testing.v1 import AppTest

TIMEOUT = 240


def fresh():
    at = AppTest.from_file("app.py", default_timeout=TIMEOUT).run()
    assert not at.exception, f"app raised on load: {[e.value for e in at.exception]}"
    return at


def run_preset(preset):
    at = fresh()
    at.selectbox[0].select(preset).run()
    assert not at.exception, f"raised selecting {preset}: {[e.value for e in at.exception]}"
    at.button[0].click().run()
    assert not at.exception, f"raised on submit: {[e.value for e in at.exception]}"
    return at


at = fresh()
print("Loaded OK.")
print("  presets:", at.selectbox[0].options)
print("  tabs:", len(at.tabs), "| selectbox:", len(at.selectbox),
      "| number_input:", len(at.number_input), "| slider:", len(at.slider))
assert len(at.tabs) == 3, f"expected 3 tabs, got {len(at.tabs)}"

print("\nSingle-student tab:")
EXPECTED = {"Thriving student": "Low", "Borderline student": "Medium",
            "Struggling student": "High"}
for preset, want in EXPECTED.items():
    a = run_preset(preset)
    md = " ".join(m.value for m in a.markdown)
    band = next((b for b in ["High risk", "Medium risk", "Low risk"] if b in md), None)
    assert band is not None, f"{preset}: no risk band rendered"
    assert band.split()[0] == want, f"{preset}: expected {want}, rendered {band}"
    assert "ffill" in md, f"{preset}: factor bars not rendered"
    assert "<svg" in md, f"{preset}: risk gauge not rendered"
    assert "Suggested next steps" in md, f"{preset}: next steps missing"
    factors = md.count('class="frow"')
    print(f"  {preset:<20} -> {band:<12} factors={factors} gauge=yes steps=yes")
    assert factors >= 5, f"{preset}: expected >=5 factor rows, got {factors}"

print("\nBatch tab:")
sample = pd.read_csv("data/sample_class.csv")
print(f"  sample register: {sample.shape[0]} students x {sample.shape[1]} cols")
at = fresh()
at.checkbox[0].check().run()
assert not at.exception, f"batch raised: {[e.value for e in at.exception]}"
assert len(at.dataframe) >= 1, "no results table rendered"
res = at.dataframe[0].value
print(f"  scored table: {res.shape[0]} rows, columns = {list(res.columns)}")
assert res.shape[0] == sample.shape[0], "row count mismatch"
assert set(res["Band"]).issubset({"Low", "Medium", "High"}), "unexpected band value"
assert res["Risk score"].is_monotonic_decreasing, "table is not sorted by risk"
print("  band counts:", res["Band"].value_counts().to_dict())
print("  top 3:", res.head(3)[["Student", "Risk score", "Band"]].to_dict("records"))

print("\nAPP END-TO-END OK")
