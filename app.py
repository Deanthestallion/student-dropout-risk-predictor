"""
Student Dropout Risk Predictor - teacher-facing Streamlit app.

Loads the pipeline saved by notebook.ipynb, scores a student (or a whole class),
and explains each score with SHAP so a teacher can see what drove it.

Run:  streamlit run app.py
"""
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

MODEL_PATH = Path(__file__).parent / "models" / "dropout_risk_model.joblib"
SAMPLE_CLASS = Path(__file__).parent / "data" / "sample_class.csv"

st.set_page_config(page_title="Student Dropout Risk Predictor",
                   page_icon="🎓", layout="wide",
                   initial_sidebar_state="expanded")

BAND_COLORS = {"Low": "#059669", "Medium": "#D97706", "High": "#DC2626"}

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
# CSS is injected with st.html, not st.markdown. st.markdown runs its input through
# a Markdown parser and HTML sanitiser, which strips the <style> tag and leaves the
# rule text visible on the page. st.html inserts raw HTML and is the supported way
# to do this. The webfont is pulled in with @import rather than a <link> tag, since
# a bare <link> is also stripped.
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  html, body, [class*="css"], .stMarkdown, .stButton, input, select, textarea {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }
  .block-container { padding-top: 3.4rem; padding-bottom: 3.5rem; max-width: 1180px; }
  #MainMenu, footer { visibility: hidden; }
  h1, h2, h3, h4 { letter-spacing: -0.02em; color: #0F172A; }

  /* Header */
  .appbar {
    background: linear-gradient(120deg, #0F766E 0%, #115E59 48%, #134E4A 100%);
    border-radius: 18px; padding: 1.5rem 1.9rem; margin-bottom: 1.4rem;
    display: flex; justify-content: space-between; align-items: center; gap: 1.5rem;
    box-shadow: 0 10px 30px rgba(15,118,110,.20);
  }
  .appbar .title { font-size: 1.6rem; font-weight: 800; color: #fff; line-height: 1.2;
                   margin: 0 0 .3rem 0; }
  .appbar .sub { font-size: .93rem; color: #CCFBF1; margin: 0; max-width: 62ch; }
  .badges { display: flex; gap: .5rem; flex-wrap: wrap; justify-content: flex-end; }
  .badge { background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.22);
           color: #F0FDFA; border-radius: 999px; padding: .3rem .8rem;
           font-size: .76rem; font-weight: 600; white-space: nowrap; }

  /* Section headings inside cards */
  .sec { display:flex; align-items:center; gap:.6rem; margin:0 0 .2rem 0; }
  .sec .n { background:#0F766E; color:#fff; width:1.5rem; height:1.5rem; border-radius:7px;
            display:inline-flex; align-items:center; justify-content:center;
            font-size:.8rem; font-weight:700; flex:none; }
  .sec .t { font-weight:700; font-size:1.02rem; color:#0F172A; }
  .sec-hint { color:#64748B; font-size:.85rem; margin:.15rem 0 .7rem 2.1rem; }

  /* Result */
  .resultcard { border-radius:16px; padding:1.3rem 1.5rem; color:#fff;
                box-shadow:0 8px 26px rgba(0,0,0,.15); }
  .resultcard .band { font-size:1.85rem; font-weight:800; line-height:1.1; }
  .resultcard .sub  { font-size:.9rem; opacity:.93; margin-top:.35rem; }

  .chip { display:inline-block; padding:.2rem .65rem; border-radius:999px;
          font-size:.78rem; font-weight:700; color:#fff; }

  /* Factor bars */
  .frow { display:grid; grid-template-columns: 1fr 132px 62px; align-items:center;
          gap:.7rem; padding:.42rem 0; border-bottom:1px solid #F1F5F9; }
  .flabel { font-size:.88rem; color:#1E293B; font-weight:500; }
  .flabel span { color:#64748B; font-weight:400; }
  .ftrack { background:#F1F5F9; border-radius:5px; height:9px; position:relative;
            overflow:hidden; }
  .ffill { position:absolute; top:0; height:100%; border-radius:5px; }
  .fval { font-size:.8rem; font-variant-numeric:tabular-nums; text-align:right;
          font-weight:600; }

  .note { background:#FFFBEB; border-left:4px solid #F59E0B; padding:.8rem 1rem;
          border-radius:9px; font-size:.86rem; color:#78350F; line-height:1.5; }
  .steps { background:#F8FAFC; border:1px solid #E2E8F0; border-radius:11px;
           padding:.9rem 1.1rem; font-size:.88rem; color:#334155; }
  .steps li { margin-bottom:.32rem; }

  div[data-testid="stMetricValue"] { font-size:1.4rem; font-weight:700; }
  div[data-testid="stMetricLabel"] { font-size:.8rem; color:#64748B; }
  .stButton button { border-radius:10px; font-weight:650; height:2.9rem; }
  .stTabs [data-baseweb="tab"] { font-weight:600; }
</style>
""")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model...")
def load_artifact():
    if not MODEL_PATH.exists():
        return None
    art = joblib.load(MODEL_PATH)
    pipe = art["pipeline"]
    pre, clf = pipe.named_steps["preprocessor"], pipe.named_steps["classifier"]
    explainer = (shap.TreeExplainer(clf) if art.get("is_tree_model")
                 else shap.LinearExplainer(clf, np.zeros((1, len(art["feature_names_encoded"])))))
    return {**art, "_pre": pre, "_clf": clf, "_explainer": explainer}


ART = load_artifact()
if ART is None:
    st.error(f"Model file not found at `{MODEL_PATH}`.\n\n"
             "Run **notebook.ipynb** end to end first. Its final section writes "
             "`models/dropout_risk_model.joblib`.")
    st.stop()

RAW_FEATURES = ART["feature_names_raw"]
LEVELS, MODES = ART["categorical_levels"], ART["categorical_modes"]
MEDIANS, RANGES = ART["numeric_medians"], ART["numeric_ranges"]
TM = ART["test_metrics_tuned"]

LABELS = {
    "G1": "First-term grade", "absences": "Days absent", "failures": "Past class failures",
    "studytime": "Weekly study time", "traveltime": "Travel time to school",
    "Medu": "Mother's education", "Fedu": "Father's education", "Mjob": "Mother's job",
    "Fjob": "Father's job", "guardian": "Guardian", "Pstatus": "Parents living together",
    "famsize": "Family size", "famrel": "Family relationship quality", "famsup": "Family support",
    "schoolsup": "Extra school support", "paid": "Paid extra classes",
    "activities": "Extra activities", "nursery": "Attended nursery",
    "higher": "Wants higher education", "internet": "Internet at home",
    "romantic": "In a relationship", "freetime": "Free time after school",
    "goout": "Going out with friends", "Dalc": "Weekday alcohol use",
    "Walc": "Weekend alcohol use", "health": "Health status", "age": "Age", "sex": "Sex",
    "school": "School", "address": "Home address type", "reason": "Reason for choosing school",
}
EDU = {0: "0 - none", 1: "1 - primary", 2: "2 - middle", 3: "3 - secondary", 4: "4 - higher"}
STUDY = {1: "1 - under 2 hrs", 2: "2 - 2 to 5 hrs", 3: "3 - 5 to 10 hrs", 4: "4 - over 10 hrs"}
TRAVEL = {1: "1 - under 15 min", 2: "2 - 15 to 30 min", 3: "3 - 30 to 60 min", 4: "4 - over 60 min"}
LH = {1: "1 - very low", 2: "2 - low", 3: "3 - average", 4: "4 - high", 5: "5 - very high"}
JOBS = {"teacher": "teacher", "health": "health care", "services": "civil service",
        "at_home": "at home", "other": "other"}
REASONS = {"home": "close to home", "reputation": "school reputation",
           "course": "course preference", "other": "other"}


def source_feature(encoded_name: str) -> str:
    """Map an encoded column back to the raw feature it came from.

    One-hot names look like 'cat__Mjob_at_home', and category values can contain
    underscores, so match against the known feature list by prefix.
    """
    n = encoded_name.replace("num__", "").replace("cat__", "")
    if n in RAW_FEATURES:
        return n
    matches = [f for f in RAW_FEATURES if n.startswith(f + "_")]
    return max(matches, key=len) if matches else n


ENC_TO_RAW = {e: source_feature(e) for e in ART["feature_names_encoded"]}


def aggregate_contributions(shap_row, row):
    """Sum SHAP values across each categorical feature's dummy columns.

    SHAP values are additive, so summing a feature's one-hot columns gives that
    feature's total contribution. The teacher sees one line per question they
    answered, labelled with the value they actually entered.
    """
    per = {}
    for enc, val in zip(ART["feature_names_encoded"], shap_row):
        raw = ENC_TO_RAW[enc]
        per[raw] = per.get(raw, 0.0) + float(val)
    s = pd.Series(per)
    return s.reindex(s.abs().sort_values(ascending=False).index)


def band_of(p: float, bands=None) -> str:
    b = bands or ART["config"]["risk_bands"]
    return "Low" if p < b["low_max"] else ("Medium" if p <= b["medium_max"] else "High")


def shap_matrix(X: pd.DataFrame) -> np.ndarray:
    sv = np.asarray(ART["_explainer"].shap_values(ART["_pre"].transform(X)))
    return sv[:, :, 1] if sv.ndim == 3 else sv


def gauge(p: float, color: str) -> str:
    r, size = 52.0, 132
    circ = 2 * math.pi * r
    return f"""
<svg viewBox="0 0 132 132" width="{size}" height="{size}" role="img"
     aria-label="Risk score {p:.0%}">
  <circle cx="66" cy="66" r="{r}" fill="none" stroke="#E2E8F0" stroke-width="13"/>
  <circle cx="66" cy="66" r="{r}" fill="none" stroke="{color}" stroke-width="13"
          stroke-linecap="round" stroke-dasharray="{circ * min(p, 1.0):.2f} {circ:.2f}"
          transform="rotate(-90 66 66)"/>
  <text x="66" y="63" text-anchor="middle" font-size="27" font-weight="800"
        fill="{color}" font-family="Inter, sans-serif">{p * 100:.0f}%</text>
  <text x="66" y="82" text-anchor="middle" font-size="10.5" fill="#64748B"
        font-family="Inter, sans-serif">risk score</text>
</svg>"""


def factor_bars(contrib: pd.Series, row: dict, n: int = 7) -> str:
    top = contrib.head(n)
    peak = max(top.abs().max(), 1e-9)
    html = []
    for feat, val in top.items():
        color = "#DC2626" if val > 0 else "#059669"
        width = abs(val) / peak * 100
        html.append(
            f'<div class="frow">'
            f'<div class="flabel">{LABELS.get(feat, feat)} <span>({row[feat]})</span></div>'
            f'<div class="ftrack"><div class="ffill" style="left:0;width:{width:.1f}%;'
            f'background:{color}"></div></div>'
            f'<div class="fval" style="color:{color}">{val:+.3f}</div>'
            f'</div>')
    return "".join(html)


NEXT_STEPS = {
    "Low": ["Continue routine monitoring, no early-warning action indicated.",
            "Re-run this check after the next set of results."],
    "Medium": ["Have a short one-to-one with the student this week.",
               "Check the attendance register for a pattern of recent absences.",
               "Confirm whether the household knows the student is falling behind.",
               "Consider a place in after-school or peer tutoring."],
    "High": ["Prioritise for support now, do not wait for end-of-term results.",
             "Arrange a meeting with the student and their guardian.",
             "Check for practical barriers: fees, distance, work at home, health.",
             "Assign a named staff member to follow up and set a review date.",
             "Record what support was offered so the outcome can be reviewed."],
}

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("#### Risk bands")
    st.caption("Move these to match the support capacity your school actually has.")
    d = ART["config"]["risk_bands"]
    low_max = st.slider("Low / Medium boundary", 0.05, 0.60, float(d["low_max"]), 0.01)
    med_max = st.slider("Medium / High boundary", 0.40, 0.95, float(d["medium_max"]), 0.01)
    if low_max >= med_max:
        st.warning("Low boundary must sit below the High boundary. Using saved defaults.")
        low_max, med_max = d["low_max"], d["medium_max"]
    BANDS = {"low_max": low_max, "medium_max": med_max}

    st.markdown(
        f"<div style='display:flex;gap:.3rem;margin-top:.5rem'>"
        f"<span class='chip' style='background:#059669'>Low &lt;{low_max:.2f}</span>"
        f"<span class='chip' style='background:#D97706'>Med</span>"
        f"<span class='chip' style='background:#DC2626'>High &gt;{med_max:.2f}</span></div>",
        unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Read this first")
    st.caption(
        "This tool predicts **end-of-year academic failure**, a documented precursor of "
        "dropout. It is not a dropout observation, and it is trained on Portuguese "
        "secondary-school data, not Nigerian data. Every score is a prompt to look closer "
        "at a student, never a verdict about them.")

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown(f"""
<div class="appbar">
  <div>
    <p class="title">Student Dropout Risk Predictor</p>
    <p class="sub">Early-warning screening for secondary schools. Turn the attendance
    register and first-term results you already collect into a ranked list of students
    who need attention this term.</p>
  </div>
  <div class="badges">
    <span class="badge">3MTT Capstone</span>
  </div>
</div>""", unsafe_allow_html=True)

tab_one, tab_class, tab_about = st.tabs(
    ["  Assess one student  ", "  Score a whole class  ", "  How it works  "])

# --------------------------------------------------------------------------- #
# Tab 1: single student
# --------------------------------------------------------------------------- #
PRESETS = {
    "Typical student (form defaults)": {},
    "Thriving student": dict(G1=16, absences=0, failures=0, studytime=4, Medu=4, Fedu=4,
                             higher="yes", goout=2, Dalc=1, Walc=1, health=5),
    "Borderline student": dict(G1=9, absences=6, failures=0, studytime=1, Medu=2, Fedu=1,
                               higher="yes"),
    "Struggling student": dict(G1=7, absences=12, failures=2, studytime=1, Medu=1, Fedu=1,
                               higher="no", schoolsup="no", goout=5, Dalc=3, Walc=4,
                               romantic="yes"),
}

with tab_one:
    c1, c2 = st.columns([2.1, 1])
    with c1:
        preset_name = st.selectbox(
            "Sample profile", list(PRESETS), index=0, label_visibility="collapsed",
            help="Pre-fills the form so you can see the tool working. Every field stays editable.")
    with c2:
        st.markdown("<div style='color:#64748B;font-size:.83rem;padding-top:.55rem'>"
                    "Pick a profile to pre-fill, or enter a real student below.</div>",
                    unsafe_allow_html=True)
    P = PRESETS[preset_name]

    def dflt(key):
        if key in P:
            return P[key]
        if key in MEDIANS:
            return int(MEDIANS[key])
        return MODES.get(key, LEVELS[key][0])

    def num(key, lo=None, hi=None, help=None):
        rlo, rhi = RANGES.get(key, [0, 100])
        return st.number_input(LABELS.get(key, key),
                               min_value=int(lo if lo is not None else rlo),
                               max_value=int(hi if hi is not None else rhi),
                               value=int(dflt(key)), step=1, help=help)

    def pick(key, scale=None, help=None):
        opts = LEVELS[key]
        d = dflt(key)
        return st.selectbox(LABELS.get(key, key), opts,
                            index=opts.index(d) if d in opts else 0,
                            format_func=(lambda v: scale.get(v, str(v))) if scale else str,
                            help=help)

    def spick(key, scale, help=None):
        lo, hi = int(RANGES[key][0]), int(RANGES[key][1])
        opts = list(range(lo, hi + 1))
        d = int(dflt(key))
        return st.selectbox(LABELS.get(key, key), opts,
                            index=opts.index(d) if d in opts else 0,
                            format_func=lambda v: scale.get(v, str(v)), help=help)

    with st.form("student_form"):
        with st.container(border=True):
            st.markdown('<div class="sec"><span class="n">1</span>'
                        '<span class="t">Academic record and attendance</span></div>'
                        '<div class="sec-hint">The two strongest inputs. Everything here is '
                        'already in your register.</div>', unsafe_allow_html=True)
            a1, a2, a3, a4 = st.columns(4)
            with a1:
                G1 = num("G1", 0, 20, "First-term grade, 0 to 20. The pass mark is 10.")
            with a2:
                absences = num("absences", 0, 93, "School days missed so far this term.")
            with a3:
                failures = num("failures", 0, 4, "Classes failed in previous years.")
            with a4:
                studytime = spick("studytime", STUDY)

            b1, b2, b3, b4 = st.columns(4)
            with b1:
                schoolsup = pick("schoolsup", help="Extra educational support from the school.")
            with b2:
                famsup = pick("famsup", help="Educational support from the family.")
            with b3:
                paid = pick("paid", help="Paid extra classes in this subject.")
            with b4:
                higher = pick("higher", help="Does the student intend to continue to higher education?")

        with st.expander("2.  Household and background  (pre-filled with typical values)"):
            st.markdown('<div class="note"><b>Use with care.</b> These are socioeconomic '
                        'proxies and are the main source of bias risk in this model. They are '
                        'here to help target support, never to lower what is expected of a '
                        'student.</div><br>', unsafe_allow_html=True)
            h1, h2, h3, h4 = st.columns(4)
            with h1:
                Medu = spick("Medu", EDU); Mjob = pick("Mjob", JOBS)
            with h2:
                Fedu = spick("Fedu", EDU); Fjob = pick("Fjob", JOBS)
            with h3:
                guardian = pick("guardian")
                Pstatus = pick("Pstatus", {"T": "living together", "A": "apart"})
            with h4:
                famsize = pick("famsize", {"LE3": "3 or fewer", "GT3": "more than 3"})
                famrel = spick("famrel", LH)
            i1, i2, i3, i4 = st.columns(4)
            with i1:
                address = pick("address", {"U": "urban", "R": "rural"})
            with i2:
                traveltime = spick("traveltime", TRAVEL)
            with i3:
                internet = pick("internet")
            with i4:
                reason = pick("reason", REASONS)

        with st.expander("3.  Personal circumstances  (pre-filled with typical values)"):
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                age = num("age", 15, 22); sex = pick("sex", {"F": "female", "M": "male"})
            with p2:
                school = pick("school", {"GP": "Gabriel Pereira", "MS": "Mousinho da Silveira"})
                health = spick("health", LH)
            with p3:
                freetime = spick("freetime", LH); goout = spick("goout", LH)
            with p4:
                Dalc = spick("Dalc", LH); Walc = spick("Walc", LH)
            q1, q2, q3, _ = st.columns(4)
            with q1:
                activities = pick("activities")
            with q2:
                nursery = pick("nursery")
            with q3:
                romantic = pick("romantic")

        submitted = st.form_submit_button("Assess dropout risk", type="primary",
                                          width="stretch")

    if submitted:
        row = {"school": school, "sex": sex, "age": age, "address": address,
               "famsize": famsize, "Pstatus": Pstatus, "Medu": Medu, "Fedu": Fedu,
               "Mjob": Mjob, "Fjob": Fjob, "reason": reason, "guardian": guardian,
               "traveltime": traveltime, "studytime": studytime, "failures": failures,
               "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
               "activities": activities, "nursery": nursery, "higher": higher,
               "internet": internet, "romantic": romantic, "famrel": famrel,
               "freetime": freetime, "goout": goout, "Dalc": Dalc, "Walc": Walc,
               "health": health, "absences": absences, "G1": G1}
        missing = [c for c in RAW_FEATURES if c not in row]
        if missing:
            st.error(f"Form is missing required inputs: {missing}")
            st.stop()

        X_one = pd.DataFrame([row])[RAW_FEATURES]
        proba = float(ART["pipeline"].predict_proba(X_one)[0, 1])
        band = band_of(proba, BANDS)
        color = BAND_COLORS[band]
        contrib = aggregate_contributions(shap_matrix(X_one)[0], row)

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2 = st.columns([1, 1.5], gap="large")

        with r1:
            with st.container(border=True):
                g1c, g2c = st.columns([1, 1.15])
                with g1c:
                    st.markdown(gauge(proba, color), unsafe_allow_html=True)
                with g2c:
                    st.markdown(
                        f"<div style='padding-top:1.4rem'>"
                        f"<div style='font-size:1.75rem;font-weight:800;color:{color};"
                        f"line-height:1.1'>{band} risk</div>"
                        f"<div style='color:#64748B;font-size:.87rem;margin-top:.35rem'>"
                        f"probability of finishing<br>the year below the pass mark</div></div>",
                        unsafe_allow_html=True)

                st.markdown(
                    f"<div class='steps'><b>Suggested next steps</b><ul style='margin:.5rem 0 0 .9rem;padding:0'>"
                    + "".join(f"<li>{s}</li>" for s in NEXT_STEPS[band])
                    + "</ul></div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='note'><b>Human in the loop.</b> This is a screening signal, "
                    "not a decision. It must never be used to stream, penalise, or exclude a "
                    "student. A teacher who knows the student decides what happens next.</div>",
                    unsafe_allow_html=True)

        with r2:
            with st.container(border=True):
                st.markdown('<div class="sec"><span class="n">?</span>'
                            '<span class="t">Why this score</span></div>'
                            '<div class="sec-hint">Each student\'s own factors, with the value '
                            'you entered in brackets. Red raised the risk, green lowered it.</div>',
                            unsafe_allow_html=True)
                st.markdown(factor_bars(contrib, row), unsafe_allow_html=True)
                st.markdown(
                    f"<div style='color:#64748B;font-size:.8rem;margin-top:.8rem'>"
                    f"Factor contributions relative to the average student. "
                    f"They sum to this student's risk score.</div>", unsafe_allow_html=True)
    else:
        st.info("Fill in the student's details above, or pick a sample profile, "
                "then select **Assess dropout risk**.")

# --------------------------------------------------------------------------- #
# Tab 2: batch
# --------------------------------------------------------------------------- #
with tab_class:
    st.markdown('<div class="sec"><span class="n">↑</span>'
                '<span class="t">Score a whole class from a register</span></div>'
                '<div class="sec-hint">Upload a CSV with one row per student. You get a ranked '
                'list, so you can start with the students who need attention most.</div>',
                unsafe_allow_html=True)

    u1, u2 = st.columns([2, 1])
    with u1:
        uploaded = st.file_uploader("Class register CSV", type=["csv"],
                                    label_visibility="collapsed")
    with u2:
        if SAMPLE_CLASS.exists():
            st.download_button("Download a sample register",
                               SAMPLE_CLASS.read_bytes(), "sample_class.csv", "text/csv",
                               width="stretch")

    use_sample = st.checkbox("Use the built-in 30-student sample register",
                             value=False, disabled=not SAMPLE_CLASS.exists())

    src = None
    if uploaded is not None:
        src = pd.read_csv(uploaded)
    elif use_sample and SAMPLE_CLASS.exists():
        src = pd.read_csv(SAMPLE_CLASS)

    if src is None:
        st.info("Upload a CSV, or tick the sample register above, to score a class.")
        with st.expander("What the CSV needs"):
            st.markdown(
                f"One row per student and these **{len(RAW_FEATURES)}** columns. An optional "
                "`student_id` column is carried through to the results.")
            st.code(", ".join(RAW_FEATURES), language="text")
    else:
        missing = [c for c in RAW_FEATURES if c not in src.columns]
        if missing:
            st.error(f"**{len(missing)} required column(s) missing:** `{'`, `'.join(missing)}`"
                     "\n\nDownload the sample register above to see the expected format.")
        else:
            ids = (src["student_id"] if "student_id" in src.columns
                   else pd.Series([f"Row {i + 1}" for i in range(len(src))]))
            Xb = src[RAW_FEATURES].copy()
            probs = ART["pipeline"].predict_proba(Xb)[:, 1]
            sv = shap_matrix(Xb)

            top_factor = []
            for i in range(len(Xb)):
                c = aggregate_contributions(sv[i], Xb.iloc[i].to_dict())
                pos = c[c > 0]
                top_factor.append(LABELS.get(pos.index[0], pos.index[0]) if len(pos) else "none")

            out = pd.DataFrame({
                "Student": ids.values,
                "Risk score": probs.round(3),
                "Band": [band_of(p, BANDS) for p in probs],
                "First-term grade": Xb["G1"].values,
                "Days absent": Xb["absences"].values,
                "Past failures": Xb["failures"].values,
                "Top risk factor": top_factor,
            }).sort_values("Risk score", ascending=False).reset_index(drop=True)

            counts = out["Band"].value_counts()
            m = st.columns(4)
            m[0].metric("Students scored", len(out))
            for i, b in enumerate(["High", "Medium", "Low"]):
                m[i + 1].metric(f"{b} risk", int(counts.get(b, 0)))

            flagged = int(counts.get("High", 0) + counts.get("Medium", 0))
            st.markdown(
                f"<div class='steps'><b>{flagged} of {len(out)} students</b> fall in the Medium "
                f"or High bands and would be reviewed first. The remaining "
                f"{len(out) - flagged} can be deprioritised this term.</div>",
                unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            st.dataframe(
                out.style
                   .map(lambda v: f"color:{BAND_COLORS.get(v, '#0F172A')};font-weight:700",
                        subset=["Band"])
                   .background_gradient(subset=["Risk score"], cmap="Reds", vmin=0, vmax=1)
                   .format({"Risk score": "{:.3f}"}),
                width="stretch", hide_index=True, height=430)

            st.download_button("Download scored register",
                               out.to_csv(index=False).encode(),
                               "scored_register.csv", "text/csv")
            st.caption("Ranked highest risk first. Open any student in the first tab to see "
                       "their full factor breakdown.")

# --------------------------------------------------------------------------- #
# Tab 3: about
# --------------------------------------------------------------------------- #
with tab_about:
    a, b = st.columns(2, gap="large")
    with a:
        with st.container(border=True):
            st.markdown("#### What this model does")
            st.markdown(f"""
It scores the probability that a student finishes the year **below the pass mark**, then
places that probability in a Low, Medium or High band.

**Target:** `{ART['target_definition']}`

A student who fails the year must repeat, and grade repetition is one of the most
consistently documented precursors of dropout in secondary education. So this finds the
population a dropout-prevention programme would want to reach.

**It does not observe dropout.** A student who leaves for reasons unrelated to grades,
such as early marriage, an income shock, or displacement, is invisible to this label.
""")
        with st.container(border=True):
            st.markdown("#### Why recall is the headline metric")
            st.markdown("""
The two errors are not equal. Missing a genuinely at-risk student means they get nothing,
and the system meant to catch them failed. Flagging a student who was fine costs a teacher
a conversation and does no harm.

So the threshold is tuned for **recall on the at-risk class**, and precision is read as the
budget constraint: how many flagged students will turn out not to have needed help.
""")
    with b:
        with st.container(border=True):
            st.markdown("#### Honest limitations")
            sw = ART.get("sensitivity_swings", {})
            abs_sw, g1_sw = sw.get("absences (days missed)"), sw.get("G1 (first-term grade)")
            att = (f"Sweeping `absences` across its whole 0 to 32 range moves the score by only "
                   f"**{abs_sw:.3f}**, while the first-term grade alone moves it by **{g1_sw:.3f}**."
                   if abs_sw is not None and g1_sw is not None else
                   "`absences` ranks 20th of 31 features by permutation importance.")
            st.markdown(f"""
- **The target is a proxy**, not observed dropout. See the panel on the left.
- **The data is Portuguese, not Nigerian.** No public student-level Nigerian
  secondary-school dataset with a dropout outcome was found. The transferable artefact is
  the pipeline, not the fitted weights. Retrain on local data before real use.
- **Attendance barely moves this model.** {att} A likely cause is that this dataset stores
  absences as one year-end total, hiding the pattern that matters: attendance that is
  *deteriorating*.
- **Small test set.** Only {int(TM['tp'] + TM['fn'])} genuinely at-risk students were held
  out, so every metric carries real uncertainty.
- **No subgroup fairness audit** was run across sex, address type, or parental education.
""")
        with st.container(border=True):
            st.markdown("#### Data protection")
            st.markdown("""
This app stores nothing. Inputs are scored in memory and gone when the page closes: no
database, no logging, no student identifiers retained. Any deployment that adds storage
would need consent, a retention policy, and access controls under the Nigeria Data
Protection Act 2023.
""")
