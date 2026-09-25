"""
Student Mark Prediction Web Dashboard
Powered by Streamlit, Scikit-learn, and Plotly.
"""

import sys
from pathlib import Path
import json

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.predict import StudentMarkPredictor, PRESET_STUDENTS, get_grade_info, generate_recommendations
from src.data_loader import load_dataset, TARGET_COL

# Page configuration
st.set_page_config(
    page_title="Student Mark Prediction AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: #c7d2fe;
        font-size: 1.05rem;
        margin: 0;
        font-weight: 400;
    }

    .metric-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.3rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }

    .prediction-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 20px;
        padding: 2.2rem;
        color: white;
        text-align: center;
        box-shadow: 0 20px 25px -5px rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1.5rem;
    }

    .score-badge {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.8rem 0;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .grade-chip {
        display: inline-block;
        padding: 0.4rem 1.4rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.15rem;
        letter-spacing: 0.5px;
        margin-top: 0.4rem;
    }

    .recs-box {
        background: #f8fafc;
        border-left: 4px solid #4f46e5;
        border-radius: 8px;
        padding: 1.2rem;
        margin-top: 1rem;
    }

    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1.4rem;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor(subject: str) -> StudentMarkPredictor:
    """Cached loader for ML predictor."""
    return StudentMarkPredictor(subject=subject)


@st.cache_data
def get_metadata() -> dict:
    """Load model training metadata."""
    meta_path = BASE_DIR / "models" / "model_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            return json.load(f)
    return {}


@st.cache_data
def get_data(subject: str) -> pd.DataFrame:
    """Cached loader for raw dataset."""
    return load_dataset(subject)


# App Header
st.markdown("""
<div class="main-header">
    <h1>🎓 Student Mark Prediction Machine Learning System</h1>
    <p>Predict final marks (G3) with high accuracy using multi-model regression pipelines trained on student performance records.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/isometric/100/graduation-cap.png", width=70)
st.sidebar.title("⚙️ Control Panel")

subject_choice = st.sidebar.selectbox(
    "Target Subject / Dataset:",
    options=["Mathematics (Math)", "Portuguese Language (Por)", "Combined Subjects"],
    index=0,
)

subject_key_map = {
    "Mathematics (Math)": "math",
    "Portuguese Language (Por)": "por",
    "Combined Subjects": "combined",
}
current_subject = subject_key_map[subject_choice]

# Load predictor
try:
    predictor = get_predictor(current_subject)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Main Tabs
tab_predict, tab_benchmark, tab_insights = st.tabs([
    "🎯 Student Mark Predictor",
    "📊 Model Performance & Leaderboard",
    "📈 Dataset Analytics",
])

# -------------------------------------------------------------
# TAB 1: PREDICTOR
# -------------------------------------------------------------
with tab_predict:
    st.subheader("Student Academic Profile & Mark Forecasting")
    st.caption("Input student historical marks, study behaviors, and background factors to calculate predicted marks.")

    # Preset Quick Loaders
    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns([1.2, 1.2, 1.2, 1.2])
    with preset_col1:
        load_top = st.button("🌟 Top Achiever", use_container_width=True)
    with preset_col2:
        load_avg = st.button("⚖️ Average Student", use_container_width=True)
    with preset_col3:
        load_risk = st.button("⚠️ At-Risk Student", use_container_width=True)
    with preset_col4:
        reset_def = st.button("🔄 Reset Defaults", use_container_width=True)

    # Determine initial values based on preset buttons
    if load_top:
        st.session_state["student_profile"] = PRESET_STUDENTS["Top Achiever (High Marks)"].copy()
    elif load_avg:
        st.session_state["student_profile"] = PRESET_STUDENTS["Average Student (Moderate Marks)"].copy()
    elif load_risk:
        st.session_state["student_profile"] = PRESET_STUDENTS["At-Risk Student (Needs Support)"].copy()
    elif reset_def or "student_profile" not in st.session_state:
        st.session_state["student_profile"] = PRESET_STUDENTS["Average Student (Moderate Marks)"].copy()

    profile = st.session_state["student_profile"]

    # Form inputs grouped logically
    col_input1, col_input2 = st.columns([1.1, 0.9])

    with col_input1:
        with st.expander("📚 1. Academic Performance & Prior Grades", expanded=True):
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                g1_val = st.slider(
                    "Period 1 Mark (G1) [0 - 20]",
                    min_value=0, max_value=20, value=int(profile.get("G1", 12)),
                    help="First period continuous assessment mark"
                )
            with sub_col2:
                g2_val = st.slider(
                    "Period 2 Mark (G2) [0 - 20]",
                    min_value=0, max_value=20, value=int(profile.get("G2", 12)),
                    help="Second period continuous assessment mark"
                )

            sub_col3, sub_col4 = st.columns(2)
            with sub_col3:
                failures_val = st.selectbox(
                    "Past Class Failures",
                    options=[0, 1, 2, 3, 4],
                    index=int(min(4, profile.get("failures", 0))),
                )
            with sub_col4:
                schoolsup_val = st.selectbox(
                    "School Extra Support",
                    options=["no", "yes"],
                    index=0 if profile.get("schoolsup", "no") == "no" else 1,
                )

        with st.expander("⏱️ 2. Study Habits & Attendance", expanded=True):
            st_col1, st_col2 = st.columns(2)
            with st_col1:
                studytime_val = st.selectbox(
                    "Weekly Study Time",
                    options=[1, 2, 3, 4],
                    format_func=lambda x: {
                        1: "1: < 2 hours / week",
                        2: "2: 2 to 5 hours / week",
                        3: "3: 5 to 10 hours / week",
                        4: "4: > 10 hours / week",
                    }[x],
                    index=int(profile.get("studytime", 2)) - 1,
                )
                traveltime_val = st.selectbox(
                    "Travel Time to School",
                    options=[1, 2, 3, 4],
                    format_func=lambda x: {
                        1: "1: < 15 min",
                        2: "2: 15 - 30 min",
                        3: "3: 30 - 60 min",
                        4: "4: > 60 min",
                    }[x],
                    index=int(profile.get("traveltime", 1)) - 1,
                )
            with st_col2:
                absences_val = st.number_input(
                    "School Absences (Days)",
                    min_value=0, max_value=93, value=int(profile.get("absences", 4)),
                )
                paid_val = st.selectbox(
                    "Extra Paid Subject Classes",
                    options=["no", "yes"],
                    index=0 if profile.get("paid", "no") == "no" else 1,
                )

        with st.expander("👨‍👩‍👧 3. Family Background & Social Support", expanded=False):
            fam_col1, fam_col2 = st.columns(2)
            with fam_col1:
                medu_val = st.selectbox(
                    "Mother's Education (Medu)",
                    options=[0, 1, 2, 3, 4],
                    format_func=lambda x: {
                        0: "0: None", 1: "1: Primary (4th grade)", 2: "2: 5th-9th grade",
                        3: "3: Secondary", 4: "4: Higher Education"
                    }[x],
                    index=int(profile.get("Medu", 2)),
                )
                mjob_val = st.selectbox(
                    "Mother's Job",
                    options=["at_home", "health", "services", "teacher", "other"],
                    index=["at_home", "health", "services", "teacher", "other"].index(profile.get("Mjob", "other")),
                )
                famsup_val = st.selectbox(
                    "Family Educational Support",
                    options=["no", "yes"],
                    index=0 if profile.get("famsup", "yes") == "no" else 1,
                )
            with fam_col2:
                fedu_val = st.selectbox(
                    "Father's Education (Fedu)",
                    options=[0, 1, 2, 3, 4],
                    format_func=lambda x: {
                        0: "0: None", 1: "1: Primary (4th grade)", 2: "2: 5th-9th grade",
                        3: "3: Secondary", 4: "4: Higher Education"
                    }[x],
                    index=int(profile.get("Fedu", 2)),
                )
                fjob_val = st.selectbox(
                    "Father's Job",
                    options=["at_home", "health", "services", "teacher", "other"],
                    index=["at_home", "health", "services", "teacher", "other"].index(profile.get("Fjob", "other")),
                )
                famrel_val = st.slider("Family Relationship Quality", 1, 5, int(profile.get("famrel", 4)))

        with st.expander("🌟 4. Student Demographics & Lifestyle", expanded=False):
            life_col1, life_col2 = st.columns(2)
            with life_col1:
                age_val = st.slider("Age", 15, 22, int(profile.get("age", 16)))
                sex_val = st.selectbox("Sex", ["F", "M"], index=0 if profile.get("sex", "F") == "F" else 1)
                higher_val = st.selectbox("Aims for Higher Education", ["yes", "no"], index=0 if profile.get("higher", "yes") == "yes" else 1)
                internet_val = st.selectbox("Internet Access at Home", ["yes", "no"], index=0 if profile.get("internet", "yes") == "yes" else 1)
            with life_col2:
                freetime_val = st.slider("Free Time After School (1-5)", 1, 5, int(profile.get("freetime", 3)))
                goout_val = st.slider("Going Out with Friends (1-5)", 1, 5, int(profile.get("goout", 3)))
                health_val = st.slider("Current Health Status (1-5)", 1, 5, int(profile.get("health", 3)))
                romantic_val = st.selectbox("In a Romantic Relationship", ["no", "yes"], index=0 if profile.get("romantic", "no") == "no" else 1)

    # Collect current student input dictionary
    current_student_data = {
        "school": profile.get("school", "GP"),
        "sex": sex_val,
        "age": age_val,
        "address": profile.get("address", "U"),
        "famsize": profile.get("famsize", "GT3"),
        "Pstatus": profile.get("Pstatus", "T"),
        "Medu": medu_val,
        "Fedu": fedu_val,
        "Mjob": mjob_val,
        "Fjob": fjob_val,
        "reason": profile.get("reason", "course"),
        "guardian": profile.get("guardian", "mother"),
        "traveltime": traveltime_val,
        "studytime": studytime_val,
        "failures": failures_val,
        "schoolsup": schoolsup_val,
        "famsup": famsup_val,
        "paid": paid_val,
        "activities": profile.get("activities", "yes"),
        "nursery": profile.get("nursery", "yes"),
        "higher": higher_val,
        "internet": internet_val,
        "romantic": romantic_val,
        "famrel": famrel_val,
        "freetime": freetime_val,
        "goout": goout_val,
        "Dalc": profile.get("Dalc", 1),
        "Walc": profile.get("Walc", 1),
        "health": health_val,
        "absences": absences_val,
        "G1": g1_val,
        "G2": g2_val,
        "subject": current_subject,
    }

    # Run inference
    pred_result = predictor.predict(current_student_data)

    # Column 2: Results Display
    with col_input2:
        st.markdown(f"""
        <div class="prediction-hero">
            <span style="text-transform: uppercase; letter-spacing: 1px; font-size: 0.85rem; color: #94a3b8; font-weight: 700;">
                Predicted Final Mark (G3)
            </span>
            <div class="score-badge">{pred_result['mark']:.2f} <span style="font-size: 1.8rem; color: #94a3b8;">/ 20</span></div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.5rem;">
                {pred_result['percentage']:.1f}% Overall Score
            </div>
            <div class="grade-chip" style="background-color: {pred_result['color']}22; color: {pred_result['color']}; border: 1.5px solid {pred_result['color']};">
                Grade {pred_result['letter_grade']} &bull; {pred_result['standing']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Plotly Gauge Indicator
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_result["mark"],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Academic Performance Scale", 'font': {'size': 14, 'color': "#475569"}},
            gauge={
                'axis': {'range': [0, 20], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                'bar': {'color': pred_result["color"]},
                'bgcolor': "white",
                'borderwidth': 1,
                'bordercolor': "#cbd5e1",
                'steps': [
                    {'range': [0, 10], 'color': '#fee2e2'},
                    {'range': [10, 14], 'color': '#fef3c7'},
                    {'range': [14, 16], 'color': '#e0f2fe'},
                    {'range': [16, 20], 'color': '#dcfce7'},
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 3},
                    'thickness': 0.75,
                    'value': 10
                }
            }
        ))
        fig_gauge.update_layout(height=210, margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Recommendations Box
        st.markdown("### 💡 AI Academic Recommendations")
        for rec in pred_result["recommendations"]:
            st.markdown(f"- {rec}")

        # Interactive "What-If" Analysis
        st.markdown("---")
        st.markdown("#### 🔮 What-If Mark Simulator")
        sim_study = st.slider("Simulate Weekly Study Time Boost:", min_value=1, max_value=4, value=max(studytime_val, 3))
        sim_abs = st.slider("Simulate Reduced Absences:", min_value=0, max_value=int(absences_val), value=min(int(absences_val), 2))
        
        sim_data = current_student_data.copy()
        sim_data["studytime"] = sim_study
        sim_data["absences"] = sim_abs
        sim_res = predictor.predict(sim_data)
        
        diff = sim_res["mark"] - pred_result["mark"]
        diff_pct = sim_res["percentage"] - pred_result["percentage"]
        sign = "+" if diff >= 0 else ""
        
        st.info(f"✨ **Simulated New Mark**: **{sim_res['mark']:.2f} / 20** ({sign}{diff:.2f} pts | {sign}{diff_pct:.1f}%) with Grade **{sim_res['letter_grade']}**")


# -------------------------------------------------------------
# TAB 2: MODEL PERFORMANCE & LEADERBOARD
# -------------------------------------------------------------
with tab_benchmark:
    st.subheader("Model Benchmarking & Evaluation Leaderboard")
    meta = get_metadata()
    current_meta = meta.get(current_subject, {})

    if current_meta:
        b_col1, b_col2, b_col3, b_col4 = st.columns(4)
        with b_col1:
            st.metric("Top Performing Algorithm", current_meta.get("model_name", "N/A"))
        with b_col2:
            st.metric("Test R² Score", f"{current_meta.get('metrics', {}).get('r2', 0.0):.4f}")
        with b_col3:
            st.metric("Root Mean Squared Error (RMSE)", f"{current_meta.get('metrics', {}).get('rmse', 0.0):.3f}")
        with b_col4:
            st.metric("Mean Absolute Error (MAE)", f"{current_meta.get('metrics', {}).get('mae', 0.0):.3f}")

        st.markdown("### 🏆 Leaderboard Comparison")
        leaderboard = current_meta.get("leaderboard", {})
        df_lb = pd.DataFrame([
            {
                "Algorithm": k,
                "Test R² Score": v["test_r2"],
                "RMSE (Lower is Better)": v["test_rmse"],
                "MAE (Mean Absolute Error)": v["test_mae"],
                "5-Fold CV R²": f"{v['cv_r2_mean']:.4f} (±{v['cv_r2_std']:.3f})",
            }
            for k, v in leaderboard.items()
        ]).sort_values(by="Test R² Score", ascending=False)

        st.dataframe(df_lb.reset_index(drop=True), use_container_width=True)

        # Plotly Leaderboard Chart
        fig_lb = px.bar(
            df_lb,
            x="Algorithm",
            y="Test R² Score",
            color="Test R² Score",
            color_continuous_scale="Viridis",
            title=f"Algorithm R² Comparison for {subject_choice}",
            text_auto=".3f",
        )
        fig_lb.update_layout(yaxis_range=[0, 1.0])
        st.plotly_chart(fig_lb, use_container_width=True)

    # Feature Importance Visualization
    st.markdown("### 🔍 Feature Importance Analysis")
    feat_img_path = BASE_DIR / "artifacts" / "feature_importance.png"
    comp_img_path = BASE_DIR / "artifacts" / "actual_vs_predicted.png"

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        if feat_img_path.exists():
            st.image(str(feat_img_path), caption="Top Influential Features (Random Forest / GBDT)", use_container_width=True)
    with img_col2:
        if comp_img_path.exists():
            st.image(str(comp_img_path), caption="Actual vs. Predicted Marks on Test Set", use_container_width=True)


# -------------------------------------------------------------
# TAB 3: DATASET INSIGHTS & EXPLORER
# -------------------------------------------------------------
with tab_insights:
    st.subheader(f"Dataset Inspection: {subject_choice}")
    df_raw = get_data(current_subject)

    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
    with d_col1:
        st.metric("Total Students Enrolled", f"{len(df_raw):,}")
    with d_col2:
        st.metric("Average Final Mark (G3)", f"{df_raw[TARGET_COL].mean():.2f} / 20")
    with d_col3:
        pass_rate = (df_raw[TARGET_COL] >= 10).mean() * 100
        st.metric("Passing Rate (≥ 10)", f"{pass_rate:.1f}%")
    with d_col4:
        st.metric("Distinction Rate (≥ 16)", f"{(df_raw[TARGET_COL] >= 16).mean() * 100:.1f}%")

    chart_c1, chart_c2 = st.columns(2)
    with chart_c1:
        fig_hist = px.histogram(
            df_raw,
            x=TARGET_COL,
            nbins=21,
            color_discrete_sequence=["#4f46e5"],
            title="Distribution of Final Marks (G3)",
            labels={TARGET_COL: "Final Mark (G3)"},
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with chart_c2:
        fig_box = px.box(
            df_raw,
            x="studytime",
            y=TARGET_COL,
            color="studytime",
            title="Final Mark (G3) vs. Weekly Study Time",
            labels={"studytime": "Weekly Study Time (1: <2h, 4: >10h)", TARGET_COL: "Final Mark"},
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Interactive Table
    st.markdown("#### 📄 Raw Dataset Explorer")
    search_term = st.text_input("Filter records by search:", "")
    if search_term:
        filtered_df = df_raw[df_raw.astype(str).apply(lambda row: row.str.contains(search_term, case=False).any(), axis=1)]
    else:
        filtered_df = df_raw
    st.dataframe(filtered_df.head(100), use_container_width=True)
