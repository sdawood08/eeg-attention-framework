# TBR1_clean_fixed_BINARY.py
# Streamlit (Teacher View) — Binary (High vs Low), leakage-safe

import json
import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from dotenv import load_dotenv
from openai import OpenAI
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier

st.set_page_config(page_title="Teacher View: EEG Attention + XAI (Binary)", layout="wide")

# ==============================
# CONFIG
# ==============================
DATA_PATH_DEFAULT = "/mnt/data/biosensor_data_with_target.csv"
DATA_PATH_WINDOWS = "biosensor_data_with_target.csv"
DATA_PATH = DATA_PATH_DEFAULT if os.path.exists(DATA_PATH_DEFAULT) else DATA_PATH_WINDOWS

KB_PATH_DEFAULT = "/mnt/data/Attention_Strategies_With_Definitions.xlsx"
KB_PATH_WINDOWS = "Attention_Strategies_With_Definitions.xlsx"

PRED_LOG_PATH_DEFAULT = "/mnt/data/predictions_log.csv"
PRED_LOG_PATH_WINDOWS = "predictions_log.csv"
PRED_LOG_PATH = PRED_LOG_PATH_DEFAULT if os.path.exists("/mnt/data") else PRED_LOG_PATH_WINDOWS

EPS = 1e-9
RANDOM_STATE = 42
TEST_SIZE = 0.20

MONASTRA = {
    (6, 11): (4.36, 5.03),
    (12, 15): (2.89, 3.31),
    (16, 20): (2.24, 2.36),
    (21, 30): (1.92, 2.13),
}

FEATURES = ["EEG_Alpha", "EEG_Beta", "EEG_Theta", "EEG_Delta", "EEG_Gamma"]

FEATURE_LABELS = {
    "EEG_Alpha": "Alpha activity",
    "EEG_Beta": "Beta activity",
    "EEG_Theta": "Theta activity",
    "EEG_Delta": "Delta activity",
    "EEG_Gamma": "Gamma activity",
}

FEATURE_MEANINGS = {
    "EEG_Beta": {
        "pos": "Higher beta-related activity (often linked to alertness and active thinking).",
        "neg": "Lower beta-related activity (often linked to reduced alertness or weaker task engagement).",
    },
    "EEG_Theta": {
        "pos": "Higher theta-related activity (often linked to mind wandering or reduced external focus).",
        "neg": "Lower theta-related activity (often linked to better external task focus).",
    },
    "EEG_Alpha": {
        "pos": "Higher alpha-related activity (often linked to relaxed state or reduced external attention).",
        "neg": "Lower alpha-related activity (often linked to greater task readiness).",
    },
    "EEG_Delta": {
        "pos": "Higher delta-related activity (may reflect fatigue/low arousal depending on context).",
        "neg": "Lower delta-related activity (less fatigue-related pattern).",
    },
    "EEG_Gamma": {
        "pos": "Higher gamma-related activity (may relate to intensive processing and integration).",
        "neg": "Lower gamma-related activity (less intensive processing signal).",
    },
}

# ----------------------------
# Load API Key
# ----------------------------
try:
    load_dotenv(r"C:\Users\sdawo\Documents\python\.env", override=True)
except Exception:
    load_dotenv(override=True)

api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
client = OpenAI(api_key=api_key) if api_key else None

# ==============================
# LABELING HELPERS (Monastra)
# ==============================
def get_thresholds(age: float):
    if pd.isna(age):
        return None
    age = float(age)
    for (lo, hi), th in MONASTRA.items():
        if lo <= age <= hi:
            return th
    return None


def monastra_attention_level(age, tbr):
    th = get_thresholds(age)
    if th is None or pd.isna(tbr):
        return np.nan
    sd1, sd15 = th
    if tbr < sd1:
        return "High"
    elif tbr < sd15:
        return "Medium"
    else:
        return "Low"


# ==============================
# DATA LOADING + FEATURE ENGINEERING
# ==============================
@st.cache_data(show_spinner=False)
def load_and_label_binary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["Age", "EEG_Theta", "EEG_Beta"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df["TBR"] = df["EEG_Theta"] / (df["EEG_Beta"] + EPS)
    df["Attention_Level_TBR"] = df.apply(
        lambda r: monastra_attention_level(r["Age"], r["TBR"]), axis=1,
    )
    df = df.dropna(subset=["Attention_Level_TBR"]).copy()
    df = df[df["Attention_Level_TBR"].isin(["High", "Low"])].copy()
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns for model: {missing}")
    df[FEATURES] = df[FEATURES].astype(float)
    df["Age"] = df["Age"].astype(float)
    df["TBR"] = df["TBR"].astype(float)
    return df


# ==============================
# MODEL TRAINING (Binary)
# ==============================
@st.cache_resource(show_spinner=False)
def train_fixed_model_binary(df_labeled: pd.DataFrame):
    assert "TBR" not in FEATURES, "TBR must never be used as a model input feature"
    X = df_labeled[FEATURES].copy()
    y = df_labeled["Attention_Level_TBR"].copy()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_enc,
    )
    model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model, le, X_train, X_test, y_train, y_test


def make_X_one(sample_row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([sample_row[FEATURES].values], columns=FEATURES).astype(float)


def compute_global_shap(model, X_data: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    shap_values_all = explainer.shap_values(X_data)
    return explainer, shap_values_all


# ==============================
# SHAP HELPERS
# ==============================
def extract_shap_vector(explainer, shap_values, X_one: pd.DataFrame, class_index: int):
    if isinstance(shap_values, list):
        sv = np.array(shap_values[class_index])[0]
        ev = explainer.expected_value
        base = float(ev[class_index]) if isinstance(ev, (list, np.ndarray)) else float(ev)
        return sv, base
    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            sv = shap_values[0, :, class_index]
        elif shap_values.ndim == 2:
            sv = shap_values[0, :]
        else:
            sv = np.zeros(X_one.shape[1])
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            base = float(ev[class_index]) if len(ev) > class_index else float(ev[0])
        else:
            base = float(ev)
        return sv, base
    if hasattr(shap_values, "values"):
        vals = np.array(shap_values.values)
        if vals.ndim == 3:
            sv = vals[0, :, class_index]
        else:
            sv = vals[0, :]
        base_vals = getattr(shap_values, "base_values", None)
        if base_vals is not None:
            base_vals = np.array(base_vals)
            base = float(base_vals[0, class_index]) if base_vals.ndim == 2 else float(base_vals[0])
        else:
            ev = explainer.expected_value
            base = float(ev[class_index]) if isinstance(ev, (list, np.ndarray)) else float(ev)
        return sv, base
    return np.zeros(X_one.shape[1]), 0.0


def get_global_importance_df(shap_values_all, feature_names):
    if isinstance(shap_values_all, list):
        arr = np.array(shap_values_all)
        mean_abs = np.mean(np.abs(arr), axis=(0, 1))
    else:
        arr = np.array(shap_values_all)
        if arr.ndim == 3:
            mean_abs = np.mean(np.abs(arr), axis=(0, 2))
        elif arr.ndim == 2:
            mean_abs = np.mean(np.abs(arr), axis=0)
        else:
            mean_abs = np.zeros(len(feature_names))
    return pd.DataFrame(
        {"Feature": feature_names, "MeanAbsSHAP": mean_abs}
    ).sort_values("MeanAbsSHAP", ascending=False)


# ==============================
# TEACHER-FRIENDLY EXPLANATIONS
# ==============================
def verbal_explanation_teacher_en(sample, pred_label, top_drivers):
    age = int(sample["Age"])
    tbr = float(sample["TBR"])
    mon = str(sample["Attention_Level_TBR"])
    lines = [
        "**Teacher Summary**",
        f"- Student age: **{age}**",
        f"- Theta/Beta Ratio (TBR): **{tbr:.2f}**",
        f"- Monastra (age-adjusted) attention label: **{mon}**",
        f"- AI predicted attention level: **{pred_label}**",
        "",
        "**Why did the AI choose this level?**",
        "The decision was mainly guided by these brain-signal patterns:",
    ]
    for feat, val in top_drivers:
        label = FEATURE_LABELS.get(feat, feat)
        impact = "supported" if float(val) > 0 else "reduced support for"
        lines.append(f"- **{label}**: this signal **{impact}** the predicted level.")
    lines.append("")
    lines.append("**Teaching Note:** The recommended strategy is chosen to match the student's current attention state.")
    return "\n".join(lines)


def cognitive_interpretation_from_shap_dynamic(pred_label: str, top_drivers: list):
    if not top_drivers:
        return (
            "**Cognitive Interpretation (SHAP-based)**\n\n"
            f"- **AI Attention Level:** **{pred_label}**\n"
            "- **Summary:** SHAP details are unavailable for this sample.\n\n"
            "**Teaching hint:** Use short tasks, chunk instructions, and quick check-ins."
        )
    dir_map = {feat: (1 if float(val) > 0 else -1) for feat, val in top_drivers}
    scores = {
        "reduced_attention": 0,
        "alert_focus": 0,
        "relaxed_state": 0,
        "fatigue_low_arousal": 0,
        "intensive_processing": 0,
    }
    if dir_map.get("EEG_Beta") == 1:
        scores["alert_focus"] += 2
    elif dir_map.get("EEG_Beta") == -1:
        scores["reduced_attention"] += 1
    if dir_map.get("EEG_Theta") == 1:
        scores["reduced_attention"] += 1
    elif dir_map.get("EEG_Theta") == -1:
        scores["alert_focus"] += 1
    if dir_map.get("EEG_Alpha") == 1:
        scores["relaxed_state"] += 1
    elif dir_map.get("EEG_Alpha") == -1:
        scores["alert_focus"] += 1
    if dir_map.get("EEG_Delta") == 1:
        scores["fatigue_low_arousal"] += 1
    if dir_map.get("EEG_Gamma") == 1:
        scores["intensive_processing"] += 1
    dominant_state = max(scores, key=scores.get)
    if pred_label == "High" and dominant_state in ["reduced_attention", "fatigue_low_arousal"]:
    dominant_state = "alert_focus"
   elif pred_label == "Low" and dominant_state == "alert_focus":
    dominant_state = "reduced_attention"
    strength = scores[dominant_state]
    if dominant_state == "reduced_attention":
        summary = "The pattern suggests reduced sustained attention and possible mind wandering."
        teaching_hint = (
            "Use short interactive prompts, chunk tasks, and quick check-ins. "
            "Try a brief attention reset (stand-stretch, 30-second recap) then re-focus the task."
        )
    elif dominant_state == "alert_focus":
        summary = "The pattern suggests strong alertness and sustained task focus."
        teaching_hint = (
            "Increase cognitive challenge using higher-order questions or problem-based activities. "
            "Ask the student to explain their reasoning aloud."
        )
    elif dominant_state == "relaxed_state":
        summary = "The pattern suggests a more relaxed state that may reduce external task focus."
        teaching_hint = (
            "Use clear goals and structured steps. Add engaging examples/visuals and frequent prompts to maintain focus."
        )
    elif dominant_state == "fatigue_low_arousal":
        summary = "The pattern may reflect low arousal/fatigue-like signals that can affect attention."
        teaching_hint = (
            "Use shorter tasks with brief breaks and varied activity formats. Reduce cognitive load and pace the lesson."
        )
    else:
        summary = "The pattern suggests stronger information processing and integration."
        teaching_hint = (
            "Use deep-learning activities: concept mapping, connecting ideas, and applied problem solving."
        )
    caution = " (This interpretation is moderate and may vary across tasks.)" if strength <= 1 else ""
    reasons = []
    for feat, val in top_drivers:
        direction = "pos" if float(val) > 0 else "neg"
        meaning = FEATURE_MEANINGS.get(feat, {}).get(direction, None)
        label = FEATURE_LABELS.get(feat, feat)
        reasons.append(
            f"- **{label}**: {meaning}"
            if meaning
            else f"- **{label}**: This signal strongly influenced the model's decision."
        )
    return (
        "**Cognitive Interpretation (SHAP-based)**\n\n"
        f"- **AI Attention Level:** **{pred_label}**\n"
        f"- **Summary:** {summary}{caution}\n\n"
        "**What signals drove this interpretation?**\n"
        + "\n".join(reasons)
        + "\n\n"
     )


# ==============================
 

# ==============================
# KB (Strategies) HELPERS
# ==============================
def resolve_kb_path():
    return KB_PATH_DEFAULT if os.path.exists(KB_PATH_DEFAULT) else KB_PATH_WINDOWS


@st.cache_data(show_spinner=False)
def load_kb(path: str) -> pd.DataFrame:
    kb = pd.read_excel(path)
    kb.columns = [c.strip() for c in kb.columns]
    if "Attention_Level" not in kb.columns:
        raise ValueError("KB must contain column: Attention_Level")
    kb["Attention_Level"] = kb["Attention_Level"].astype(str).str.strip()
    return kb


def filter_kb_by_level(kb: pd.DataFrame, pred_label: str) -> pd.DataFrame:
    kb2 = kb.copy()
    kb2["Attention_Level"] = kb2["Attention_Level"].astype(str).str.strip().str.lower()
    lvl = str(pred_label).strip().lower()
    return kb2[kb2["Attention_Level"] == lvl].copy()


def kb_candidates_for_llm(kb_level: pd.DataFrame, max_items: int = 12):
    def get_col(row, options):
        for c in options:
            if c in row and pd.notna(row[c]):
                return str(row[c])
        return ""
    items = []
    for _, r in kb_level.head(max_items).iterrows():
        items.append({
            "strategy": get_col(r, ["Strategy", "Name", "Title", "Strategy_Name"]),
            "definition": get_col(r, ["Definition", "Description", "Details"]),
            "reference": get_col(r, ["Reference", "Citation", "Source"]),
        })
    return [it for it in items if it["strategy"].strip() != ""]


def build_llm_prompt(pred_label: str, top_drivers: list, cognitive_text: str, candidates: list):
    if not top_drivers:
        top_drivers = [("EEG_Beta", 0.0)]
    shap_summary = [
        {"feature": f, "direction": ("positive" if float(v) > 0 else "negative"), "shap": float(v)}
        for f, v in top_drivers
    ]
    return f"""
You are an educational strategy selector.

CONSTRAINTS:
- Select exactly ONE strategy from the candidates list.
- Do NOT invent new strategies.
- Output MUST be valid JSON only.
- For every reason you give, include a short supporting snippet from the KB definition/description as "evidence_from_kb".
  Use ONLY the candidate's provided definition/description/reference as evidence.

INPUTS:
Predicted attention level: {pred_label}
SHAP top drivers: {json.dumps(shap_summary, ensure_ascii=False)}
Cognitive interpretation: {cognitive_text}

Candidates (each has strategy, definition, reference):
{json.dumps(candidates, ensure_ascii=False)}

Return JSON ONLY in this exact structure:
{{
  "selected_strategy": "<must match exactly one candidate strategy>",
  "why_best": [
    {{"reason": "<reason 1>", "evidence_from_kb": "<short phrase from candidate definition>"}},
    {{"reason": "<reason 2>", "evidence_from_kb": "<short phrase from candidate definition>"}},
    {{"reason": "<reason 3>", "evidence_from_kb": "<short phrase from candidate definition>"}}
  ],
  "how_to_apply": [
    {{"step": "<step 1>", "evidence_from_kb": "<short phrase if applicable>"}},
    {{"step": "<step 2>", "evidence_from_kb": "<short phrase if applicable>"}},
    {{"step": "<step 3>", "evidence_from_kb": "<short phrase if applicable>"}}
  ]
}}
""".strip()


def llm_choose_strategy(client: OpenAI, prompt: str, model_name="gpt-4o-mini"):
    resp = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=600,
    )
    txt = (resp.choices[0].message.content or "").strip()
    try:
        return json.loads(txt)
    except Exception:
        s, e = txt.find("{"), txt.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(txt[s: e + 1])
        return {}


def validate_llm_choice(result: dict, candidates: list):
    allowed = {c["strategy"] for c in candidates}
    chosen = str(result.get("selected_strategy", "")).strip()
    return chosen if chosen in allowed else None


def find_kb_row_by_strategy_name(kb_level: pd.DataFrame, selected_name: str):
    for col in ["Strategy", "Name", "Title", "Strategy_Name"]:
        if col in kb_level.columns:
            hit = kb_level[kb_level[col].astype(str).str.strip() == selected_name]
            if not hit.empty:
                return hit.iloc[0]
    return None


# ==============================
# LOGGING
# ==============================
def append_prediction_log(path, sample_row, pred_label, top_drivers, chosen_strategy=None, model_name="GradientBoosting"):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    drivers_txt = "; ".join([f"{f}:{float(v):+.4f}" for f, v in top_drivers]) if top_drivers else ""
    strat_name = ""
    if chosen_strategy is not None and isinstance(chosen_strategy, (pd.Series, dict)):
        for col in ["Strategy", "Strategy_Name", "Name", "Title"]:
            if col in chosen_strategy and pd.notna(chosen_strategy[col]):
                strat_name = str(chosen_strategy[col])
                break
    row = {
        "timestamp": now,
        "Age": float(sample_row["Age"]),
        "TBR": float(sample_row["TBR"]),
        "Monastra_Label": str(sample_row["Attention_Level_TBR"]),
        "Model_Prediction": str(pred_label),
        "Top_SHAP_Drivers": drivers_txt,
        "Recommended_Strategy": strat_name,
        "model": model_name,
        "EEG_Alpha": float(sample_row["EEG_Alpha"]),
        "EEG_Beta": float(sample_row["EEG_Beta"]),
        "EEG_Theta": float(sample_row["EEG_Theta"]),
        "EEG_Delta": float(sample_row["EEG_Delta"]),
        "EEG_Gamma": float(sample_row["EEG_Gamma"]),
    }
    out = pd.DataFrame([row])
    if os.path.exists(path):
        out.to_csv(path, mode="a", index=False, header=False)
    else:
        out.to_csv(path, mode="w", index=False, header=True)


# ==============================
# UI (Teacher only)
# ==============================
st.title("EEG + XAI (Teacher View)")

# Load data + train model
try:
    df_labeled = load_and_label_binary(DATA_PATH)
    if df_labeled.empty:
        st.error("No samples available after binary filtering (High/Low). Check Age ranges and thresholds.")
        st.stop()
    model, le, X_train, X_test, y_train, y_test = train_fixed_model_binary(df_labeled)
    global_explainer, shap_values_all = compute_global_shap(model, X_train)
except Exception as e:
    st.error(f"Setup failed: {e}")
    st.stop()

# Teacher selects ONE sample
st.sidebar.header("Student Selection")
row_id = st.sidebar.slider("Sample/Student Number", 0, len(df_labeled) - 1, 0, 1)

sample = df_labeled.iloc[int(row_id)]
X_one = make_X_one(sample)

pred_class = int(model.predict(X_one)[0])
pred_label = le.inverse_transform([pred_class])[0]

# 1) Summary table
st.subheader("Selected student data")
summary = pd.DataFrame([{
    "Age": int(sample["Age"]),
    "EEG_Alpha": float(sample["EEG_Alpha"]),
    "EEG_Beta": float(sample["EEG_Beta"]),
    "EEG_Theta": float(sample["EEG_Theta"]),
    "EEG_Delta": float(sample["EEG_Delta"]),
    "EEG_Gamma": float(sample["EEG_Gamma"]),
    "TBR (for labeling/display)": float(sample["TBR"]),
    "Attention (Monastra, binary)": sample["Attention_Level_TBR"],
    "Model Prediction (EEG-only)": pred_label,
}])
st.dataframe(summary, use_container_width=True)

# 2) Global SHAP
st.subheader("🌍 Global SHAP Importance")
try:
    global_importance_df = get_global_importance_df(shap_values_all, FEATURES)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        fig_global_bar, ax_global_bar = plt.subplots(figsize=(5, 3))
        ax_global_bar.barh(
            global_importance_df["Feature"][::-1],
            global_importance_df["MeanAbsSHAP"][::-1],
        )
        ax_global_bar.set_xlabel("Mean |SHAP value|")
        ax_global_bar.set_ylabel("Feature")
        ax_global_bar.set_title("(a) Global Feature Importance")
        st.pyplot(fig_global_bar, clear_figure=True)
    with col2:
        st.dataframe(global_importance_df, use_container_width=True)

    # Beeswarm plot
    st.markdown("**SHAP Beeswarm Plot — Feature Direction and Distribution**")
    st.caption(
        "Each dot = one sample. Color = feature value (red = high, blue = low). "
        "Position on x-axis = direction of impact on prediction."
    )

    shap_arr = np.array(shap_values_all)
    if shap_arr.ndim == 3:
        sv_for_beeswarm = shap_arr[:, :, 1]
    elif shap_arr.ndim == 2:
        sv_for_beeswarm = shap_arr
    else:
        sv_for_beeswarm = shap_arr

    shap.summary_plot(
        sv_for_beeswarm,
        X_train,
        feature_names=FEATURES,
        plot_type="dot",
        show=False,
        color_bar=True,
        alpha=0.8,
    )
    plt.gcf().set_size_inches(9, 4)
    plt.tight_layout()
    plt.savefig(
        "shap_beeswarm_final.png",
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        format='png',
    )
    fig_beeswarm = plt.gcf()
    st.pyplot(fig_beeswarm, clear_figure=True)

except Exception as e:
    st.warning(f"Global SHAP rendering failed: {e}")

# 3) Local SHAP + Cognitive Interpretation
st.subheader("🔍 SHAP (Why did the model predict this?)")

top_drivers = []
cog_text_md = ""
used_llm = False

try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_one)

    sv, base = extract_shap_vector(explainer, shap_values, X_one, pred_class)

    feat_names = FEATURES[:]
    x_vals = X_one.iloc[0].values
    min_len = min(len(feat_names), len(sv), len(x_vals))
    feat_names, sv, x_vals = feat_names[:min_len], sv[:min_len], x_vals[:min_len]

    contrib = pd.DataFrame({
        "Feature": feat_names,
        "Feature_Value": x_vals,
        "SHAP_value": sv,
        "Abs_SHAP": np.abs(sv),
    }).sort_values("Abs_SHAP", ascending=False)

    st.subheader("📊 Local SHAP Feature Importance")
    fig_local_bar, ax_local_bar = plt.subplots(figsize=(8, 4))
    local_plot_df = contrib.head(5).iloc[::-1]
    ax_local_bar.barh(local_plot_df["Feature"], local_plot_df["SHAP_value"])
    ax_local_bar.set_xlabel("SHAP value")
    ax_local_bar.set_ylabel("Feature")
    ax_local_bar.set_title("Selected Student: Top SHAP Contributions")
    st.pyplot(fig_local_bar, clear_figure=True)

    st.subheader("Top Drivers (Most Influential EEG Features)")
    st.dataframe(contrib[["Feature", "Feature_Value", "SHAP_value"]].head(6), use_container_width=True)

    top_drivers = list(contrib[["Feature", "SHAP_value"]].head(3).itertuples(index=False, name=None))

    st.subheader("🧩 SHAP Waterfall Plot (Selected Student)")
    explanation = shap.Explanation(
        values=sv,
        base_values=base,
        data=X_one.iloc[0].values,
        feature_names=feat_names,
    )
    shap.plots.waterfall(explanation, max_display=5, show=False)
    fig_waterfall = plt.gcf()
    st.pyplot(fig_waterfall, clear_figure=True)

   
    st.subheader("🧠 Cognitive Interpretation (SHAP-based)")
    cog_text_md = cognitive_interpretation_from_shap_dynamic(pred_label, top_drivers)
    st.markdown(cog_text_md)
    if used_llm:
        st.caption("✨ Enhanced with AI for better teacher clarity")

except Exception as e:
    st.warning(f"SHAP rendering failed: {e}")
    cog_text_md = (
        "**Cognitive Interpretation (SHAP-based)**\n\n"
        f"- **AI Attention Level:** **{pred_label}**\n"
        "- **Summary:** SHAP details are unavailable for this sample.\n\n"
        "**Teaching hint:** Use short tasks, chunk instructions, and quick check-ins."
    )
    st.markdown(cog_text_md)

# 4) Strategy recommendation (KB + optional LLM)
st.subheader("🎯 Recommended Teaching Strategy")
kb_path = resolve_kb_path()
chosen = None
llm_result = None
kb_level = pd.DataFrame()

try:
    kb = load_kb(kb_path)
    kb_level = filter_kb_by_level(kb, pred_label)

    if kb_level.empty:
        st.info("No matching strategy found for this attention level in the knowledge base.")
        chosen = None
    else:
        candidates = kb_candidates_for_llm(kb_level, max_items=12)

        if len(candidates) == 0:
            st.warning("No clear strategy names in KB. Using first row as fallback.")
            chosen = kb_level.iloc[0]
        elif client is None:
            st.info("LLM not active (no OPENAI_API_KEY). Using first KB strategy as fallback.")
            chosen = kb_level.iloc[0]
        else:
            prompt = build_llm_prompt(pred_label, top_drivers, cog_text_md, candidates)
            llm_result = llm_choose_strategy(client, prompt, model_name="gpt-4o-mini")
            selected_name = validate_llm_choice(llm_result, candidates)

            if selected_name is None:
                st.warning("LLM output invalid. Falling back to first KB strategy.")
                chosen = kb_level.iloc[0]
            else:
                st.success(f"LLM Selected Strategy: {selected_name}")

                st.markdown("### Why best (with KB evidence)")
                for i, item in enumerate(llm_result.get("why_best", []), start=1):
                    reason = item.get("reason", "")
                     
                     
                st.markdown("### How to apply (with KB evidence)")
                for i, item in enumerate(llm_result.get("how_to_apply", []), start=1):
                    step = item.get("step", "")
                    
                    

                chosen = find_kb_row_by_strategy_name(kb_level, selected_name)
                if chosen is None:
                    chosen = kb_level.iloc[0]

    if chosen is not None:
        strategy_name = ""
        for col in ["Strategy", "Strategy_Name", "Name", "Title"]:
            if col in chosen and pd.notna(chosen[col]):
                strategy_name = str(chosen[col]).strip()
                break
        definition = ""
        for col in ["Definition", "Description", "Details"]:
            if col in chosen and pd.notna(chosen[col]):
                definition = str(chosen[col]).strip()
                break
        reference = ""
        for col in ["Reference", "Citation", "Source"]:
            if col in chosen and pd.notna(chosen[col]):
                reference = str(chosen[col]).strip()
                break
        st.markdown(f"**Strategy:** {strategy_name if strategy_name else '(No clear strategy name in KB)'}")
        if definition:
            st.markdown(f"**Definition:** {definition}")
        if reference:
            st.markdown(f"**Reference:** {reference}")

except Exception as e_kb:
    chosen = None
    st.warning(f"Strategy KB failed: {e_kb}")

# 5) Logging
if chosen is None and not kb_level.empty:
    chosen = kb_level.iloc[0]

try:
    append_prediction_log(
        PRED_LOG_PATH,
        sample,
        pred_label,
        top_drivers,
        chosen_strategy=chosen,
        model_name="GradientBoosting",
    )
except Exception:
    pass
