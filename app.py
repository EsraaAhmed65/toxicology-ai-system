import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Toxicology AI Detection System",
    page_icon="🧪",
    layout="wide"
)

calibrated_model = joblib.load("calibrated_random_forest_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")


def predict_case(case_df):
    probabilities = calibrated_model.predict_proba(case_df)[0]
    classes = calibrated_model.classes_

    sorted_indices = np.argsort(probabilities)[::-1]
    top1_idx = sorted_indices[0]
    top2_idx = sorted_indices[1]

    top1_prob = probabilities[top1_idx]
    top2_prob = probabilities[top2_idx]

    prediction = classes[top1_idx]
    second_possible = classes[top2_idx]
    margin = (top1_prob - top2_prob) * 100
    confidence = top1_prob * 100

    review_flag = (
        prediction == "Borderline"
        or (
            second_possible == "Borderline"
            and margin < 20
        )
        or (
            confidence < 70
            and margin < 15
        )
    )

    if prediction == "Borderline" and margin < 30:
        confidence_text = "⚠️ Borderline likely"
    elif review_flag:
        confidence_text = "⚠️ Review recommended"
    elif confidence >= 85:
        confidence_text = "High confidence prediction"
    elif confidence >= 70:
        confidence_text = "Moderate confidence prediction"
    else:
        confidence_text = "Low confidence prediction"

    return {
        "prediction": prediction,
        "second_possible": second_possible,
        "confidence": float(round(confidence, 2)),
        "margin_top1_top2": float(round(margin, 2)),
        "probabilities": probabilities,
        "confidence_text": confidence_text,
        "review_flag": review_flag
    }


def generate_explanation(input_data, prediction, second_possible):
    reasons_for_prediction = []
    reasons_for_uncertainty = []

    if input_data["radiation_pattern"] == 1:
        if prediction in ["Polonium-like", "Borderline"]:
            reasons_for_prediction.append(
                "Radiation pattern is present, which supports a radiation-like or borderline toxic profile."
            )
        else:
            reasons_for_uncertainty.append(
                "Radiation pattern is present, which may conflict with a non-radiation class."
            )

    if input_data["feed_exposure"] == 1:
        if prediction in ["Ricin-like", "Borderline"]:
            reasons_for_prediction.append(
                "Feed exposure is present, which supports a feed-borne toxic pattern."
            )
        else:
            reasons_for_uncertainty.append(
                "Feed exposure is present, which may pull the case toward a toxic exposure pattern."
            )

    if input_data["sudden_onset"] == 1:
        if prediction in ["Ricin-like", "Borderline"]:
            reasons_for_prediction.append(
                "Sudden onset supports an acute toxic presentation."
            )
        else:
            reasons_for_uncertainty.append(
                "Sudden onset is unusual for a stable non-toxic pattern."
            )

    if input_data["progressive_deterioration"] == 1:
        if prediction in ["Polonium-like", "Borderline"]:
            reasons_for_prediction.append(
                "Progressive deterioration supports an evolving systemic toxic pattern."
            )
        else:
            reasons_for_uncertainty.append(
                "Progressive deterioration may indicate a more severe pattern than the main prediction."
            )

    if input_data["gi_damage_score"] >= 2:
        if prediction in ["Ricin-like", "Borderline"]:
            reasons_for_prediction.append(
                "Moderate to high GI damage supports a gastrointestinal toxic pattern."
            )
        else:
            reasons_for_uncertainty.append(
                "GI damage is notable and may suggest a more toxic class."
            )

    if input_data["cell_damage_score"] >= 2:
        reasons_for_prediction.append(
            "Cell damage score is elevated, which supports a toxic injury pattern."
        )

    if input_data["multi_organ_damage"] >= 2:
        reasons_for_prediction.append(
            "Multi-organ damage is present, which increases the suspicion of systemic toxicity."
        )

    if input_data["marrow_suppression"] >= 2:
        if prediction in ["Polonium-like", "Borderline"]:
            reasons_for_prediction.append(
                "Marrow suppression is elevated, which is important in radiation-like or mixed toxic patterns."
            )
        else:
            reasons_for_uncertainty.append(
                "Marrow suppression is higher than expected for the current class."
            )

    if input_data["renal_injury_score"] >= 2 or input_data["hepatic_injury_score"] >= 2:
        reasons_for_prediction.append(
            "Organ injury severity is moderate to high, supporting a clinically significant case."
        )

    if input_data["time_to_onset_hours"] <= 12:
        if prediction in ["Ricin-like", "Borderline"]:
            reasons_for_prediction.append(
                "Rapid symptom onset supports an acute exposure scenario."
            )
        else:
            reasons_for_uncertainty.append(
                "Rapid onset may be more consistent with an acute toxic class."
            )

    if input_data["time_to_onset_hours"] >= 72:
        if prediction in ["Polonium-like", "Borderline", "Normal"]:
            reasons_for_prediction.append(
                "Later onset is compatible with slower or progressive patterns."
            )
        else:
            reasons_for_uncertainty.append(
                "Late onset may conflict with a very acute toxic profile."
            )

    if second_possible != prediction:
        reasons_for_uncertainty.append(
            f"The case also shares features with {second_possible}."
        )

    return reasons_for_prediction, reasons_for_uncertainty


def generate_alerts(input_data, prediction, second_possible, confidence, margin, review_flag):
    alerts = []

    if review_flag:
        alerts.append("Review recommended: this case may be borderline, uncertain, or mixed.")

    if prediction == "Borderline" and second_possible in ["Ricin-like", "Polonium-like"]:
        alerts.append(f"Mixed toxic pattern suspected: the case overlaps with {second_possible}.")

    if prediction == "Normal" and (
        input_data["cell_damage_score"] >= 2
        or input_data["multi_organ_damage"] >= 2
        or input_data["marrow_suppression"] >= 2
    ):
        alerts.append("Possible false reassurance: predicted Normal despite notable damage-related findings.")

    if input_data["radiation_pattern"] == 1 and input_data["feed_exposure"] == 1:
        alerts.append("Conflicting exposure signals detected: both radiation-like and feed-related clues are present.")

    if prediction == "Polonium-like" and input_data["radiation_pattern"] == 0:
        alerts.append("Prediction-pattern mismatch: Polonium-like predicted without a strong radiation clue.")

    if prediction == "Ricin-like" and input_data["feed_exposure"] == 0 and input_data["gi_damage_score"] < 2:
        alerts.append("Prediction-pattern mismatch: Ricin-like predicted with limited feed/GI support.")

    if confidence >= 85 and second_possible == "Borderline":
        alerts.append("High-confidence result still overlaps with Borderline. Interpret with caution.")

    if margin < 15:
        alerts.append("Low separation between top predictions: decision boundary is tight.")

    return alerts


st.markdown("""
<style>
[data-testid="stHeader"] {display: none;}
[data-testid="stToolbar"] {display: none;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1180px;
}

html, body, [class*="css"] {
    font-family: "Segoe UI", sans-serif;
}

.main-title {
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.05;
    margin-bottom: 0.35rem;
    color: #F8FAFC;
}

.sub-title {
    font-size: 1.05rem;
    color: #A7B4C7;
    margin-bottom: 1rem;
}

.cool-divider {
    height: 2px;
    border: 0;
    background: linear-gradient(
        90deg,
        rgba(56,189,248,0.0) 0%,
        rgba(56,189,248,0.9) 25%,
        rgba(16,185,129,0.9) 75%,
        rgba(16,185,129,0.0) 100%
    );
    margin: 1rem 0 1.4rem 0;
}

.section-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 22px;
    padding: 22px 22px 16px 22px;
    margin-bottom: 1.2rem;
}

.section-title {
    font-size: 1.35rem;
    font-weight: 750;
    margin-bottom: 0.9rem;
    color: #F8FAFC;
}

.feature-box {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 14px 14px 8px 14px;
    margin-bottom: 12px;
}

.feature-name {
    font-size: 1rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 0.15rem;
}

.feature-help {
    font-size: 0.88rem;
    color: #93A4B8;
    margin-bottom: 0.45rem;
    line-height: 1.35;
}

.stRadio > div {
    gap: 0.7rem;
}

.stRadio label {
    font-weight: 600 !important;
}

.stButton > button {
    width: 100%;
    border-radius: 14px;
    padding: 0.9rem 1rem;
    font-size: 1rem;
    font-weight: 700;
    border: 1px solid rgba(255,255,255,0.12);
    background: linear-gradient(90deg, rgba(37,99,235,0.22), rgba(16,185,129,0.22));
}

.summary-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.95), rgba(15,23,42,0.98));
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 22px;
    padding: 22px;
    min-height: 180px;
}

.summary-label {
    color: #94A3B8;
    font-size: 0.92rem;
    margin-bottom: 0.35rem;
}

.summary-value {
    font-size: 2rem;
    font-weight: 800;
    color: #F8FAFC;
    margin-bottom: 0.35rem;
}

.confidence-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #F8FAFC;
    margin-bottom: 0.35rem;
}

.result-note {
    color: #9FB0C4;
    font-size: 0.92rem;
    line-height: 1.4;
    margin-bottom: 0.2rem;
}

.badge-high, .badge-medium, .badge-low {
    display: inline-block;
    margin-top: 0.8rem;
    padding: 8px 14px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.95rem;
    width: fit-content;
}

.badge-high {
    background: rgba(16,185,129,0.15);
    color: #4ADE80;
}

.badge-medium {
    background: rgba(245,158,11,0.16);
    color: #FBBF24;
}

.badge-low {
    background: rgba(239,68,68,0.16);
    color: #F87171;
}

.prob-section-title {
    font-size: 1.35rem;
    font-weight: 750;
    margin: 1.7rem 0 1rem 0;
    color: #F8FAFC;
}

.prob-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 18px;
    padding: 16px;
    margin-bottom: 14px;
}

.prob-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.7rem;
}

.prob-name {
    font-size: 1rem;
    font-weight: 700;
    color: #F8FAFC;
}

.prob-percent {
    font-size: 1rem;
    font-weight: 800;
    color: #DCE7F5;
}

.prob-track {
    width: 100%;
    height: 10px;
    background: rgba(255,255,255,0.08);
    border-radius: 999px;
    overflow: hidden;
}

.prob-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #38BDF8, #3B82F6);
}

.small-note {
    color: #94A3B8;
    font-size: 0.88rem;
    margin-top: 0.5rem;
}

.helper-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 18px;
    margin-top: 1rem;
}

.helper-title {
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 0.7rem;
    color: #F8FAFC;
}

.helper-text {
    color: #A7B4C7;
    line-height: 1.5;
    font-size: 0.96rem;
}

.stAlert {
    border-radius: 14px;
    margin-top: 1rem;
    margin-bottom: 1.3rem;
}

.reset-btn .stButton > button {
    width: auto !important;
    min-width: 110px;
    padding: 0.55rem 0.9rem !important;
    font-size: 0.92rem !important;
    border-radius: 12px !important;
}

.explain-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 18px;
    margin-top: 1rem;
}

.explain-title {
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
    color: #F8FAFC;
}

.explain-list {
    color: #C7D3E0;
    font-size: 0.95rem;
    line-height: 1.6;
    margin: 0;
    padding-left: 1.2rem;
}

.alert-card {
    background: rgba(245,158,11,0.10);
    border: 1px solid rgba(245,158,11,0.28);
    border-radius: 16px;
    padding: 16px;
    margin-top: 1rem;
}

.alert-title {
    font-size: 1.08rem;
    font-weight: 700;
    color: #FBBF24;
    margin-bottom: 0.65rem;
}

.alert-list {
    color: #F8E7B0;
    font-size: 0.95rem;
    line-height: 1.6;
    margin: 0;
    padding-left: 1.2rem;
}

@media (max-width: 900px) {
    .summary-card {
        min-height: auto;
    }
}
</style>
""", unsafe_allow_html=True)

pretty_names = {
    "fever": "Fever",
    "vomiting": "Vomiting",
    "diarrhea": "Diarrhea",
    "anorexia": "Anorexia",
    "dehydration": "Dehydration",
    "sudden_onset": "Sudden Onset",
    "progressive_deterioration": "Progressive Deterioration",
    "feed_exposure": "Feed Exposure",
    "radiation_pattern": "Radiation Pattern",
    "herd_cluster": "Herd Cluster",
    "weakness_score": "Weakness Score",
    "cell_damage_score": "Cell Damage Score",
    "gi_damage_score": "GI Damage Score",
    "multi_organ_damage": "Multi-Organ Damage",
    "marrow_suppression": "Marrow Suppression",
    "renal_injury_score": "Renal Injury Score",
    "hepatic_injury_score": "Hepatic Injury Score",
    "time_to_onset_hours": "Time to Onset (Hours)"
}

feature_help = {
    "fever": "Elevated body temperature.",
    "vomiting": "Presence of vomiting.",
    "diarrhea": "Presence of diarrhea.",
    "anorexia": "Loss of appetite.",
    "dehydration": "Signs of reduced body fluids.",
    "sudden_onset": "Symptoms started suddenly.",
    "progressive_deterioration": "Condition worsened gradually over time.",
    "feed_exposure": "Possible exposure through contaminated feed.",
    "radiation_pattern": "Signs suggesting radiation-like toxic injury.",
    "herd_cluster": "More than one animal affected in the same setting.",
    "weakness_score": "Severity from 0 to 3.",
    "cell_damage_score": "Overall cellular damage severity from 0 to 3.",
    "gi_damage_score": "Gastrointestinal damage severity from 0 to 3.",
    "multi_organ_damage": "Damage affecting multiple organs.",
    "marrow_suppression": "Severity of bone marrow suppression.",
    "renal_injury_score": "Kidney injury severity from 0 to 3.",
    "hepatic_injury_score": "Liver injury severity from 0 to 3.",
    "time_to_onset_hours": "Estimated hours between exposure and symptom onset."
}

binary_features = [
    "fever",
    "vomiting",
    "diarrhea",
    "anorexia",
    "dehydration",
    "sudden_onset",
    "progressive_deterioration",
    "feed_exposure",
    "radiation_pattern",
    "herd_cluster"
]

score_features = [
    "weakness_score",
    "cell_damage_score",
    "gi_damage_score",
    "multi_organ_damage",
    "marrow_suppression",
    "renal_injury_score",
    "hepatic_injury_score"
]

time_feature = "time_to_onset_hours"

example_cases = {
    "Manual Entry": None,
    "Example: Normal": {
        "fever": 0, "vomiting": 0, "diarrhea": 0, "anorexia": 0, "dehydration": 0,
        "sudden_onset": 0, "progressive_deterioration": 0, "feed_exposure": 0,
        "radiation_pattern": 0, "herd_cluster": 0,
        "weakness_score": 0, "cell_damage_score": 0, "gi_damage_score": 0,
        "multi_organ_damage": 0, "marrow_suppression": 0,
        "renal_injury_score": 0, "hepatic_injury_score": 0,
        "time_to_onset_hours": 48.0
    },
    "Example: Ricin-like": {
        "fever": 1, "vomiting": 1, "diarrhea": 1, "anorexia": 1, "dehydration": 1,
        "sudden_onset": 1, "progressive_deterioration": 0, "feed_exposure": 1,
        "radiation_pattern": 0, "herd_cluster": 1,
        "weakness_score": 3, "cell_damage_score": 3, "gi_damage_score": 3,
        "multi_organ_damage": 2, "marrow_suppression": 1,
        "renal_injury_score": 2, "hepatic_injury_score": 3,
        "time_to_onset_hours": 6.0
    },
    "Example: Polonium-like": {
        "fever": 0, "vomiting": 0, "diarrhea": 0, "anorexia": 1, "dehydration": 0,
        "sudden_onset": 0, "progressive_deterioration": 1, "feed_exposure": 0,
        "radiation_pattern": 1, "herd_cluster": 0,
        "weakness_score": 3, "cell_damage_score": 3, "gi_damage_score": 1,
        "multi_organ_damage": 3, "marrow_suppression": 3,
        "renal_injury_score": 3, "hepatic_injury_score": 2,
        "time_to_onset_hours": 96.0
    },
    "Example: Borderline": {
        "fever": 1, "vomiting": 0, "diarrhea": 1, "anorexia": 1, "dehydration": 1,
        "sudden_onset": 1, "progressive_deterioration": 1, "feed_exposure": 0,
        "radiation_pattern": 1, "herd_cluster": 0,
        "weakness_score": 2, "cell_damage_score": 2, "gi_damage_score": 2,
        "multi_organ_damage": 2, "marrow_suppression": 1,
        "renal_injury_score": 2, "hepatic_injury_score": 1,
        "time_to_onset_hours": 30.0
    }
}


def load_example_to_session(example_name):
    example_data = example_cases.get(example_name)

    if example_data is None:
        for feature in binary_features:
            st.session_state[feature] = "No"

        for feature in score_features:
            st.session_state[feature] = 0

        st.session_state[time_feature] = 24.0
        return

    for feature in binary_features:
        st.session_state[feature] = "Yes" if example_data[feature] == 1 else "No"

    for feature in score_features:
        st.session_state[feature] = int(example_data[feature])

    st.session_state[time_feature] = float(example_data[time_feature])


if "selected_example" not in st.session_state:
    st.session_state.selected_example = "Manual Entry"
    load_example_to_session("Manual Entry")

st.markdown('<div class="main-title">🧪 Toxicology AI Detection System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">AI-based decision support tool for classifying livestock toxicology cases into Normal, Ricin-like, Polonium-like, or Borderline patterns.</div>',
    unsafe_allow_html=True
)
st.markdown('<div class="cool-divider"></div>', unsafe_allow_html=True)

top_left, top_right = st.columns([1.15, 0.35])

with top_left:
    selected_example = st.selectbox(
        "Example Cases",
        options=list(example_cases.keys()),
        index=list(example_cases.keys()).index(st.session_state.selected_example)
    )

    if selected_example != st.session_state.selected_example:
        st.session_state.selected_example = selected_example
        load_example_to_session(selected_example)
        st.rerun()

with top_right:
    st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
    if st.button("Reset"):
        st.session_state.selected_example = "Manual Entry"
        load_example_to_session("Manual Entry")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

current_values = example_cases.get(st.session_state.selected_example, None)
input_data = {}

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Clinical & Exposure Features</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")

    for i, feature in enumerate(binary_features):
        with (c1 if i % 2 == 0 else c2):
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.markdown(f'<div class="feature-name">{pretty_names[feature]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="feature-help">{feature_help[feature]}</div>', unsafe_allow_html=True)

            choice = st.radio(
                label=pretty_names[feature],
                options=["No", "Yes"],
                horizontal=True,
                label_visibility="collapsed",
                key=feature
            )
            input_data[feature] = 1 if choice == "Yes" else 0
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Severity Scores & Timing</div>', unsafe_allow_html=True)

    for feature in score_features:
        st.markdown(f'<div class="feature-name">{pretty_names[feature]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="feature-help">{feature_help[feature]}</div>', unsafe_allow_html=True)
        input_data[feature] = st.slider(
            label=pretty_names[feature],
            min_value=0,
            max_value=3,
            step=1,
            label_visibility="collapsed",
            key=feature
        )

    st.markdown(f'<div class="feature-name">{pretty_names[time_feature]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="feature-help">{feature_help[time_feature]}</div>', unsafe_allow_html=True)
    input_data[time_feature] = st.number_input(
        label=pretty_names[time_feature],
        min_value=0.0,
        max_value=200.0,
        step=1.0,
        label_visibility="collapsed",
        key=time_feature
    )

    st.markdown('<div class="small-note">Severity scores range from 0 (none) to 3 (severe).</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

predict_clicked = st.button("Run Prediction")

if predict_clicked:
    input_df = pd.DataFrame([input_data])[feature_columns]
    result = predict_case(input_df)

    prediction = result["prediction"]
    second_possible = result["second_possible"]
    confidence = result["confidence"]
    margin = result["margin_top1_top2"]
    probabilities = result["probabilities"]
    confidence_text = result["confidence_text"]
    review_flag = result["review_flag"]

    reasons_for_prediction, reasons_for_uncertainty = generate_explanation(
        input_data, prediction, second_possible
    )

    alerts = generate_alerts(
        input_data, prediction, second_possible, confidence, margin, review_flag
    )

    class_labels = list(calibrated_model.classes_)
    prob_df = pd.DataFrame({
        "Class": class_labels,
        "Probability": probabilities * 100
    }).sort_values(by="Probability", ascending=False)

    if review_flag:
        badge_class = "badge-medium"
    elif confidence >= 85:
        badge_class = "badge-high"
    elif confidence >= 70:
        badge_class = "badge-medium"
    else:
        badge_class = "badge-low"

    st.markdown('<div class="cool-divider"></div>', unsafe_allow_html=True)

    res1, res2 = st.columns(2, gap="large")

    with res1:
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-label">Predicted Class</div>
            <div class="summary-value">{prediction}</div>
            <div class="{badge_class}">{confidence_text}</div>
        </div>
        """, unsafe_allow_html=True)

    with res2:
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-label">Confidence</div>
            <div class="confidence-value">{confidence:.2f}%</div>
            <div class="result-note">Second possible: <b>{second_possible}</b></div>
            <div class="result-note">Top-2 margin: <b>{margin:.2f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

    if prediction == "Ricin-like":
        action_text = "Suggested action: investigate recent feed exposure, isolate suspicious sources, and prioritize toxicological review."
    elif prediction == "Polonium-like":
        action_text = "Suggested action: review progressive systemic damage pattern and investigate radiation-like toxic exposure urgently."
    elif prediction == "Borderline":
        action_text = "Suggested action: treat as uncertain or mixed toxic pattern and perform additional investigation before a final conclusion."
    elif review_flag:
        action_text = "Suggested action: this case may overlap with a borderline or mixed toxic pattern. Additional review is recommended before final interpretation."
    else:
        action_text = "Suggested action: findings are more consistent with a non-severe pattern, but continue routine monitoring if clinically needed."

    st.markdown(f"""
    <div class="helper-card">
        <div class="helper-title">Recommended Action</div>
        <div class="helper-text">{action_text}</div>
    </div>
    """, unsafe_allow_html=True)

    prediction_html = "".join([f"<li>{reason}</li>" for reason in reasons_for_prediction]) if reasons_for_prediction else "<li>No dominant explanation extracted.</li>"
    uncertainty_html = "".join([f"<li>{reason}</li>" for reason in reasons_for_uncertainty]) if reasons_for_uncertainty else "<li>No major uncertainty driver detected.</li>"

    st.markdown(f"""
    <div class="explain-card">
        <div class="explain-title">Why this prediction?</div>
        <ul class="explain-list">
            {prediction_html}
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="explain-card">
        <div class="explain-title">What may reduce certainty?</div>
        <ul class="explain-list">
            {uncertainty_html}
        </ul>
    </div>
    """, unsafe_allow_html=True)

    if review_flag:
        st.warning(
            f"This case may be uncertain, borderline, or mixed. "
            f"Predicted: {prediction} | Second possible: {second_possible} | Margin: {margin:.2f}%"
        )
    else:
        st.success("Prediction generated successfully.")

    if alerts:
        alerts_html = "".join([f"<li>{alert}</li>" for alert in alerts])

        st.markdown(f"""
        <div class="alert-card">
            <div class="alert-title">Risk Alerts</div>
            <ul class="alert-list">
                {alerts_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="prob-section-title">Probability Breakdown</div>', unsafe_allow_html=True)

    for _, row in prob_df.iterrows():
        width = max(float(row["Probability"]), 1.5)
        st.markdown(f"""
        <div class="prob-card">
            <div class="prob-top">
                <div class="prob-name">{row["Class"]}</div>
                <div class="prob-percent">{row["Probability"]:.2f}%</div>
            </div>
            <div class="prob-track">
                <div class="prob-fill" style="width:{width}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)