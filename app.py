"""
SpamShield - decision-support console for an enterprise mail gateway.

Loads the five pre-trained classifiers from /model (no retraining happens here),
scores an uploaded hold-out CSV, and layers gateway-policy tooling on top.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report, roc_curve)

from spamshield_features import (FEATURE_NAMES, TARGET, log1p_transform,  # noqa: F401
                                 extract_features_from_text)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
PALETTE = {"ham": "#2E7D6B", "spam": "#C8553D", "ink": "#1F2A37", "mute": "#8A94A6",
           "accent": "#3B5BA9"}

st.set_page_config(page_title="SpamShield | Mail Gateway Console", page_icon="🛡️",
                   layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
.kpi {background:#F5F7FA; border:1px solid #E3E8EF; border-radius:10px; padding:12px 14px;}
.kpi .l {font-size:.72rem; letter-spacing:.06em; text-transform:uppercase; color:#6B7686;}
.kpi .v {font-size:1.55rem; font-weight:650; color:#1F2A37; line-height:1.2;}
.verdict {border-radius:12px; padding:16px 20px; font-size:1.25rem; font-weight:650;}
.verdict.spam {background:#FBEAE6; color:#9A3420; border:1px solid #F0C4BA;}
.verdict.ham {background:#E4F3EE; color:#1E6552; border:1px solid #B9DFD3;}
.verdict.mid {background:#FFF4DB; color:#8A5A00; border:1px solid #F1DBA0;}
@media (prefers-color-scheme: dark) {
 .kpi {background:#1E2530; border-color:#2E3846;} .kpi .v {color:#F0F3F8;} .kpi .l{color:#9AA5B5;}
}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- resources
@st.cache_resource(show_spinner=False)
def load_models():
    return {n: joblib.load(MODEL_DIR / f) for n, f in MODEL_FILES.items()}


@st.cache_data(show_spinner=False)
def load_artifacts():
    meta = json.load(open(MODEL_DIR / "meta.json"))
    stored = json.load(open(MODEL_DIR / "metrics.json"))
    return meta, stored


@st.cache_data(show_spinner=False)
def bundled_test():
    return pd.read_csv(ROOT / "test_data.csv")


MODELS = load_models()
META, STORED = load_artifacts()


def kpi(label, value):
    st.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div></div>',
                unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def score_all(df_features: pd.DataFrame) -> dict:
    """Spam probability from every saved model (inference only)."""
    return {n: m.predict_proba(df_features[FEATURE_NAMES])[:, 1] for n, m in MODELS.items()}


def metric_row(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return {"Accuracy": accuracy_score(y, pred), "AUC": roc_auc_score(y, proba),
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0), "MCC": matthews_corrcoef(y, pred)}


# ------------------------------------------------------------------- sidebar
st.sidebar.title("🛡️ SpamShield")
st.sidebar.caption("Mail-gateway triage console · UCI Spambase · 5 classifiers")
PAGES = ["Mission Control", "Model Evaluation", "Model Arena",
         "Gateway Policy Simulator", "Live Email Inspector", "Insights & Caveats"]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.markdown("**Test data**")
up = st.sidebar.file_uploader("Upload test CSV (57 features + `is_spam`)", type="csv")
data_note = ""
if up is not None:
    try:
        df = pd.read_csv(up)
        data_note = f"Using **{up.name}**"
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(f"Could not read CSV: {e}")
        df = bundled_test()
        data_note = "Upload failed - using bundled sample"
else:
    df = bundled_test()
    data_note = "No upload - using bundled 921-email hold-out set"
st.sidebar.caption(data_note)

missing = [c for c in FEATURE_NAMES + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Uploaded file is missing {len(missing)} required column(s): "
             f"{', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}. "
             f"It must contain all 57 feature columns and the `{TARGET}` target column.")
    st.stop()
df = df.dropna(subset=FEATURE_NAMES + [TARGET]).reset_index(drop=True)
if not set(df[TARGET].unique()) <= {0, 1} or df[TARGET].nunique() < 2:
    st.error(f"`{TARGET}` must contain both classes (0 = legitimate, 1 = spam).")
    st.stop()
y = df[TARGET].astype(int).values
SCORES = score_all(df)


# =========================================================== 1. MISSION CONTROL
if page == "Mission Control":
    st.title("SpamShield · Mail Gateway Console")
    st.markdown(
        "An inbound mail gateway has to make three calls on every message - **deliver**, "
        "**quarantine**, or **block** - and the two ways to be wrong are *not* equally "
        "expensive. A blocked invoice from a customer costs far more than one stray "
        "newsletter reaching an inbox. SpamShield compares five classic classifiers on "
        "the UCI Spambase corpus and turns their probabilities into an operating policy.")
    d = META["dataset"]
    c = st.columns(5)
    with c[0]: kpi("Emails (full corpus)", f"{d['rows']:,}")
    with c[1]: kpi("Features", d["features"])
    with c[2]: kpi("Spam share", f"{d['spam'] / d['rows']:.1%}")
    with c[3]: kpi("Train / Test", f"{d['train_rows']:,} / {d['test_rows']:,}")
    with c[4]: kpi("Best model (MCC)", META["winner"].replace("Support Vector Machine", "SVM"))

    st.subheader("Leaderboard on the loaded test set")
    board = pd.DataFrame({n: metric_row(y, s) for n, s in SCORES.items()}).T
    board = board.sort_values("MCC", ascending=False)
    st.dataframe(board.style.format("{:.4f}").background_gradient(cmap="Greens", axis=0),
                 height=235)

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("How the pipeline works")
        st.markdown(
            "1. **Features** - 48 word-frequency %, 6 punctuation-frequency %, 3 capital-letter "
            "run statistics per email.\n"
            "2. **Pre-processing** - `log1p` to tame heavy tails, then standardisation "
            "(trees use raw values).\n"
            "3. **Models** - tuned with 5-fold stratified grid search on F1, trained on 80 % of "
            "the data, frozen to `.joblib`.\n"
            "4. **This app** - loads the frozen pipelines and only runs *inference*; "
            "nothing is retrained on uploaded data.")
        st.info("Use the sidebar to open the policy simulator (cost-aware thresholds) or the "
                "live inspector (paste a raw email and see why it scores the way it does).")
    with right:
        st.subheader("Class balance (loaded data)")
        fig, ax = plt.subplots(figsize=(4.6, 2.6))
        counts = [(y == 0).sum(), (y == 1).sum()]
        ax.barh(["Legitimate", "Spam"], counts, color=[PALETTE["ham"], PALETTE["spam"]])
        for i, v in enumerate(counts):
            ax.text(v, i, f" {v:,}", va="center", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, max(counts) * 1.18)
        st.pyplot(fig)

# ========================================================== 2. MODEL EVALUATION
elif page == "Model Evaluation":
    st.title("Model Evaluation")
    st.caption("Pick a saved model; metrics are computed on the loaded test CSV "
               "using the frozen pipeline (no retraining).")
    c1, c2 = st.columns([2, 1])
    with c1:
        name = st.selectbox("Model", list(MODEL_FILES), index=list(MODEL_FILES).index(META["winner"]))
    with c2:
        thr = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05,
                        help="Probability above which an email is labelled spam.")
    proba = SCORES[name]
    m = metric_row(y, proba, thr)
    cols = st.columns(6)
    for col, (k, v) in zip(cols, m.items()):
        with col: kpi(k, f"{v:.4f}")
    st.write("")

    pred = (proba >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    left, mid, right = st.columns([1, 1, 1])
    with left:
        st.markdown("**Confusion matrix**")
        fig, ax = plt.subplots(figsize=(4, 3.4))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=14,
                    color="white" if v > cm.max() / 2 else PALETTE["ink"])
        ax.set_xticks([0, 1], ["Ham", "Spam"]); ax.set_yticks([0, 1], ["Ham", "Spam"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"{fp} legitimate emails wrongly flagged · {fn} spam emails slipped through")
    with mid:
        st.markdown("**ROC curve**")
        fpr, tpr, _ = roc_curve(y, proba)
        fig, ax = plt.subplots(figsize=(4, 3.4))
        ax.plot(fpr, tpr, color=PALETTE["accent"], lw=2, label=f"AUC = {m['AUC']:.3f}")
        ax.plot([0, 1], [0, 1], "--", color=PALETTE["mute"], lw=1)
        ax.scatter([fp / (fp + tn)], [tp / (tp + fn)], color=PALETTE["spam"], zorder=5,
                   label=f"threshold {thr:.2f}")
        ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate")
        ax.legend(loc="lower right", frameon=False); ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
    with right:
        st.markdown("**Score distribution**")
        fig, ax = plt.subplots(figsize=(4, 3.4))
        bins = np.linspace(0, 1, 26)
        ax.hist(proba[y == 0], bins, alpha=.75, color=PALETTE["ham"], label="Legitimate")
        ax.hist(proba[y == 1], bins, alpha=.75, color=PALETTE["spam"], label="Spam")
        ax.axvline(thr, color=PALETTE["ink"], ls="--", lw=1)
        ax.set_xlabel("P(spam)"); ax.set_yscale("log"); ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)

    st.markdown("**Classification report**")
    rep = pd.DataFrame(classification_report(
        y, pred, target_names=["Legitimate", "Spam"], output_dict=True, zero_division=0)).T
    rep["support"] = rep["support"].astype(int)
    st.dataframe(rep.style.format({"precision": "{:.3f}", "recall": "{:.3f}",
                                   "f1-score": "{:.3f}"}))
    info = STORED["extras"][name]
    st.caption(f"Tuned hyper-parameters: `{info['best_params']}` · 5-fold CV F1 "
               f"{info['cv_f1_mean']:.4f} ± {info['cv_f1_std']:.4f}")

# ================================================================ 3. MODEL ARENA
elif page == "Model Arena":
    st.title("Model Arena")
    st.caption("All five frozen models, same emails, side by side.")
    board = pd.DataFrame({n: metric_row(y, s) for n, s in SCORES.items()}).T
    st.dataframe(board.style.format("{:.4f}").highlight_max(axis=0, color="#CFE8DF"))

    left, right = st.columns(2)
    with left:
        st.markdown("**Metric comparison**")
        fig, ax = plt.subplots(figsize=(6, 3.6))
        w = 0.16; xs = np.arange(len(board.columns))
        for i, (n, row) in enumerate(board.iterrows()):
            ax.bar(xs + (i - 2) * w, row.values, w, label=n)
        ax.set_xticks(xs, board.columns); ax.set_ylim(0.6, 1.0)
        ax.legend(fontsize=7, ncol=2, frameon=False, loc="lower left")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
    with right:
        st.markdown("**ROC curves**")
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for n, s in SCORES.items():
            f, t, _ = roc_curve(y, s)
            ax.plot(f, t, lw=1.8, label=f"{n} ({roc_auc_score(y, s):.3f})")
        ax.plot([0, 1], [0, 1], "--", color=PALETTE["mute"], lw=1)
        ax.set_xlim(0, 0.35); ax.set_ylim(0.6, 1.0)
        ax.set_xlabel("False-positive rate (zoomed)"); ax.set_ylabel("True-positive rate")
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)

    st.subheader("Where do the models disagree?")
    votes = np.column_stack([(s >= 0.5).astype(int) for s in SCORES.values()])
    n_spam_votes = votes.sum(axis=1)
    ens = (n_spam_votes >= 3).astype(int)
    mrow = metric_row(y, n_spam_votes / 5.0, 0.5)
    unanimous = ((n_spam_votes == 0) | (n_spam_votes == 5)).mean()
    a, b, c = st.columns(3)
    with a: kpi("Unanimous verdicts", f"{unanimous:.1%}")
    with b: kpi("Majority-vote accuracy", f"{accuracy_score(y, ens):.4f}")
    with c: kpi("Majority-vote MCC", f"{matthews_corrcoef(y, ens):.4f}")
    contested = df.assign(spam_votes=n_spam_votes,
                          **{f"p_{n.split()[0].lower()}": s.round(3) for n, s in SCORES.items()})
    contested = contested[(n_spam_votes > 0) & (n_spam_votes < 5)]
    contested["truth"] = np.where(contested[TARGET] == 1, "spam", "legit")
    show = ["truth", "spam_votes"] + [c_ for c_ in contested.columns if c_.startswith("p_")]
    st.write(f"**{len(contested)}** emails split the panel - the best candidates for human review:")
    st.dataframe(contested[show].sort_values("spam_votes", ascending=False).head(60))

# ====================================================== 4. GATEWAY POLICY SIMULATOR
elif page == "Gateway Policy Simulator":
    st.title("Gateway Policy Simulator")
    st.markdown("A gateway doesn't have to pick one cut-off. Use **two thresholds**: below *deliver* "
                "mail is delivered, above *block* it is rejected, and everything in between goes to "
                "a **quarantine** queue for review. Set what each mistake costs and the simulator "
                "finds the cheapest policy.")
    name = st.selectbox("Scoring model", list(MODEL_FILES),
                        index=list(MODEL_FILES).index(META["winner"]))
    s = SCORES[name]

    c1, c2, c3, c4 = st.columns(4)
    with c1: c_fp = st.number_input("Cost: legit mail BLOCKED (₹)", 0.0, 100000.0, 500.0, 50.0,
                                    help="Lost invoice, missed customer reply, escalation effort.")
    with c2: c_fn = st.number_input("Cost: spam DELIVERED (₹)", 0.0, 100000.0, 40.0, 5.0,
                                    help="Distraction, phishing exposure risk, clean-up.")
    with c3: c_rv = st.number_input("Cost: quarantine review (₹)", 0.0, 10000.0, 6.0, 1.0,
                                    help="Admin / user time to inspect one quarantined message.")
    with c4: volume = st.number_input("Inbound mails per day", 100, 5_000_000, 50_000, 1000)

    grid = np.round(np.arange(0.0, 1.0001, 0.02), 2)

    def policy_cost(lo, hi):
        blocked = s >= hi
        delivered = s < lo
        quar = ~blocked & ~delivered
        fp_b = ((y == 0) & blocked).sum(); fn_d = ((y == 1) & delivered).sum(); q = quar.sum()
        return (fp_b * c_fp + fn_d * c_fn + q * c_rv) / len(y), fp_b, fn_d, q

    best = (np.inf, None)
    for lo in grid:
        for hi in grid[grid >= lo]:
            cst = policy_cost(lo, hi)[0]
            if cst < best[0]: best = (cst, (lo, hi))
    single = {t: policy_cost(t, t)[0] for t in grid}
    t_single = min(single, key=single.get)

    st.markdown("##### Your policy")
    p1, p2 = st.columns(2)
    with p1: lo = st.slider("Deliver below", 0.0, 1.0, float(best[1][0]), 0.02)
    with p2: hi = st.slider("Block at or above", 0.0, 1.0, float(max(best[1][1], lo)), 0.02)
    if hi < lo:
        st.warning("'Block' threshold must be ≥ 'Deliver' threshold."); st.stop()

    cost, fp_b, fn_d, q = policy_cost(lo, hi)
    daily = cost * volume
    ref_cost = policy_cost(0.5, 0.5)[0] * volume
    k = st.columns(5)
    with k[0]: kpi("Cost / day", f"₹{daily:,.0f}")
    with k[1]: kpi("vs. plain 0.5 cut-off", f"{(daily - ref_cost) / ref_cost:+.0%}" if ref_cost else "n/a")
    with k[2]: kpi("Legit blocked / day", f"{fp_b / len(y) * volume:,.0f}")
    with k[3]: kpi("Spam delivered / day", f"{fn_d / len(y) * volume:,.0f}")
    with k[4]: kpi("Quarantine queue / day", f"{q / len(y) * volume:,.0f}")

    st.write("")
    left, right = st.columns(2)
    with left:
        st.markdown("**Cost vs. single cut-off**")
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.plot(list(single), [v * volume for v in single.values()], color=PALETTE["accent"], lw=2)
        ax.axvline(t_single, color=PALETTE["mute"], ls=":", lw=1)
        ax.axhline(best[0] * volume, color=PALETTE["ham"], ls="--", lw=1.4,
                   label=f"best two-threshold policy ₹{best[0] * volume:,.0f}")
        ax.set_xlabel("single threshold"); ax.set_ylabel("₹ / day")
        ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
    with right:
        st.markdown("**Where the three actions fall**")
        fig, ax = plt.subplots(figsize=(6, 3.4))
        bins = np.linspace(0, 1, 41)
        ax.hist(s[y == 0], bins, alpha=.75, color=PALETTE["ham"], label="Legitimate")
        ax.hist(s[y == 1], bins, alpha=.75, color=PALETTE["spam"], label="Spam")
        ax.axvspan(lo, hi, color="#F2C14E", alpha=.28, label="quarantine band")
        ax.set_yscale("log"); ax.set_xlabel("P(spam)"); ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)

    st.success(f"Cheapest policy for these costs: **deliver < {best[1][0]:.2f}**, "
               f"**block ≥ {best[1][1]:.2f}** → ₹{best[0] * volume:,.0f}/day. "
               f"Best single cut-off is {t_single:.2f} → ₹{single[t_single] * volume:,.0f}/day.")

# ======================================================= 5. LIVE EMAIL INSPECTOR
elif page == "Live Email Inspector":
    st.title("Live Email Inspector")
    st.caption("Paste a raw email. The app derives the same 57 Spambase features and scores it with "
               "every saved model, then explains the winner's call.")
    SAMPLES = {
        "Prize / lottery pitch": (
            "CONGRATULATIONS!!! You have been selected to receive a FREE $1000 cash prize!\n\n"
            "Dear winner, our records show that your email address was chosen in this month's "
            "promotion. To claim your money, click here NOW and confirm your details. This is a "
            "limited offer and you must order your free gift today. There is no business "
            "obligation and your credit is guaranteed - just reply to this email, send your "
            "address and we will deliver your cash prize within days. Hurry, many people have "
            "already received their money and this offer will not last. Remember, it costs you "
            "nothing: the prize is completely FREE, and you can receive even more by sharing the "
            "offer with friends. To remove your name from our list, reply with REMOVE in the "
            "subject. Act now, because FREE money like this will not come again!!! $$$"),
        "Project status mail": (
            "Hi team,\n\nRe: project meeting on Thursday. I have attached the report for the "
            "conference and the original data table for review. Please send me your comments "
            "before the meeting so we can finalise the schedule. Thanks, George"),
        "Invoice reminder": (
            "Dear customer, your account statement for October is ready. Please review the "
            "invoice details and confirm payment of the pending amount by the due date. If you "
            "have already paid, kindly ignore this message. Regards, Accounts Team"),
    }
    choice = st.selectbox("Load a sample (or write your own below)", ["-"] + list(SAMPLES))
    text = st.text_area("Raw email text", SAMPLES.get(choice, ""), height=190)

    if text.strip():
        feats = extract_features_from_text(text)
        x = pd.DataFrame([feats])[FEATURE_NAMES]
        probs = {n: float(m.predict_proba(x)[0, 1]) for n, m in MODELS.items()}
        focus = st.selectbox("Explain model", list(MODEL_FILES),
                             index=list(MODEL_FILES).index(META["winner"]))
        p = probs[focus]
        cls, msg = (("spam", "BLOCK · likely spam") if p >= 0.7 else
                    ("ham", "DELIVER · looks legitimate") if p < 0.3 else
                    ("mid", "QUARANTINE · uncertain, send for review"))
        st.markdown(f'<div class="verdict {cls}">{msg} &nbsp;·&nbsp; P(spam) = {p:.1%} '
                    f'<span style="font-weight:400;font-size:.9rem">({focus})</span></div>',
                    unsafe_allow_html=True)
        _mean = (pd.Series(META["ham_mean"]) * META["dataset"]["ham"]
                 + pd.Series(META["spam_mean"]) * META["dataset"]["spam"]) / META["dataset"]["rows"]
        _z = ((x.iloc[0] - _mean).abs() / pd.Series(META["feature_std"]).replace(0, 1))
        if _z.max() > 8:
            st.warning(f"Unusual input: `{_z.idxmax()}` is {_z.max():.0f}σ from the training average "
                       "(typical of very short emails). Distance-based models - especially the RBF "
                       "SVM and KNN - get less reliable far outside the training range, so compare "
                       "the five models rather than trusting one score.")
        st.write("")
        left, right = st.columns([1, 1.25])
        with left:
            st.markdown("**All five models**")
            fig, ax = plt.subplots(figsize=(4.6, 3))
            names = list(probs)
            ax.barh(names, [probs[n] for n in names],
                    color=[PALETTE["spam"] if probs[n] >= 0.5 else PALETTE["ham"] for n in names])
            ax.axvline(0.5, color=PALETTE["ink"], ls="--", lw=1)
            ax.set_xlim(0, 1); ax.invert_yaxis(); ax.set_xlabel("P(spam)")
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig)
        with right:
            st.markdown("**Why? (occlusion test)**")
            ham_mean = pd.Series(META["ham_mean"])
            active = [f for f in FEATURE_NAMES if abs(feats[f] - ham_mean[f]) > 1e-9]
            variants = []
            for f in active:
                v = x.copy(); v[f] = ham_mean[f]; variants.append(v)
            if variants:
                pv = MODELS[focus].predict_proba(pd.concat(variants))[:, 1]
                eff = pd.Series(p - pv, index=active).sort_values()
                top = pd.concat([eff.head(4), eff.tail(6)]).drop_duplicates()
                top = top[top.abs() > 1e-4]
                fig, ax = plt.subplots(figsize=(5.4, 3))
                ax.barh([f.replace("word_freq_", "word: ").replace("char_freq_", "char: ")
                         .replace("capital_run_length_", "caps ") for f in top.index], top.values,
                        color=[PALETTE["spam"] if v > 0 else PALETTE["ham"] for v in top.values])
                ax.axvline(0, color=PALETTE["ink"], lw=1)
                ax.set_xlabel("change in P(spam) when the feature is reset to a typical legit value")
                ax.xaxis.label.set_size(8); ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig)
            else:
                st.info("Every feature already matches a typical legitimate email.")
        with st.expander("Extracted feature vector (57 values)"):
            st.dataframe(x.T.rename(columns={0: "value"}))
        st.caption("Heads-up: Spambase dates from 1999 and the extractor here is a faithful-but-approximate "
                   "re-implementation, so treat verdicts on modern phishing as illustrative.")

# ==================================================== 6. INSIGHTS & CAVEATS
else:
    st.title("Insights & Caveats")
    left, right = st.columns(2)
    with left:
        st.subheader("What drives the winning model")
        imp = pd.Series(META["permutation_importance"]).sort_values()
        fig, ax = plt.subplots(figsize=(5.6, 4.6))
        ax.barh(imp.index, imp.values, color=PALETTE["accent"])
        ax.set_xlabel("AUC drop when feature is shuffled")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
    with right:
        st.subheader("Typical value: legit vs spam")
        top_feats = [f for f in META["permutation_importance"]][:8]
        comp = pd.DataFrame({"Legitimate": [META["ham_mean"][f] for f in top_feats],
                             "Spam": [META["spam_mean"][f] for f in top_feats]}, index=top_feats)
        st.dataframe(comp.style.format("{:.3f}").background_gradient(cmap="OrRd", axis=1))
        st.caption("Mean feature value per class in the training split.")

    st.subheader("Cross-validation & tuned settings")
    rows = [{"Model": n, "CV F1 (mean)": e["cv_f1_mean"], "CV F1 (std)": e["cv_f1_std"],
             "Best parameters": str(e["best_params"])} for n, e in STORED["extras"].items()]
    st.dataframe(pd.DataFrame(rows).set_index("Model").style.format(
        {"CV F1 (mean)": "{:.4f}", "CV F1 (std)": "{:.4f}"}))

    st.subheader("Read before deploying anywhere real")
    st.markdown(
        "- **Dataset bias:** the strongest 'legitimate' signals (`george`, `hp`, `edu`, `650`, `meeting`) "
        "come from one HP Labs mailbox in the late 1990s. Models learn that *this* inbox's personal "
        "vocabulary means ham - a different company would need retraining on its own mail.\n"
        "- **Concept drift:** modern phishing is polite, brand-accurate and low on `!`/`$`. Frequency "
        "features alone will under-detect it.\n"
        "- **Out-of-range inputs:** the RBF-SVM falls back towards its baseline for points far from any "
        "training email (e.g. a 30-word email where 7 % of words are 'free'), while linear/tree models "
        "keep extrapolating. The Live Inspector warns when an input is that extreme.\n"
        "- **Bag-of-words only:** no headers, SPF/DKIM results, URLs or attachments are used.\n"
        "- **Costs are assumptions:** the policy simulator is only as good as the rupee figures you enter.")
