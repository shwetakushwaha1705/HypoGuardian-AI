"""HypoGuardian AI: Hypoglycemia Monitoring Using Advanced Deep Learning.

Main Streamlit Web Application providing interactive medical analytics,
data upload validation, time-series visualization, LSTM early-warning inference,
prediction history auditing, model evaluation dashboards, and viva preparation.
"""

import sys
from pathlib import Path

# Ensure project root directory is in sys.path for Streamlit Cloud and Linux environments
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datetime import datetime
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    DATA_SAMPLE_DIR,
    DEFAULT_CLASSIFICATION_THRESHOLD,
    EVALUATION_RESULTS_PATH,
    HYPO_THRESHOLD_MG_DL,
    METADATA_PATH,
    MODEL_PATH,
    MODELS_DIR,
    SCALER_PATH,
    SEQUENCE_LENGTH,
)
from src.data_loader import (
    DataIngestionError,
    MissingColumnsError,
    load_glucose_csv,
)
from src.prediction import (
    ArtifactsNotFoundError,
    HypoPredictionEngine,
    InvalidSequenceError,
)
from src.preprocessing import preprocess_glucose_data
from utils.visualization import (
    create_cgm_trend_chart,
    create_confusion_matrix_chart,
    create_prediction_window_chart,
    create_time_in_range_pie,
    create_training_curves_chart,
)

# ==========================================
# 1. STREAMLIT APP CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="HypoGuardian AI | Hypoglycemia Early Warning",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern medical dashboard aesthetic
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Metrics card styling */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 2px;
    }
    
    /* Risk Badge */
    .risk-badge-high {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #f87171;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .risk-badge-low {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #4ade80;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Clinical Disclaimer */
    .disclaimer-banner {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 4px;
        color: #92400e;
        font-size: 0.86rem;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 2. SESSION STATE INITIALIZATION
# ==========================================
if "current_df" not in st.session_state:
    st.session_state.current_df = None

if "audit_report" not in st.session_state:
    st.session_state.audit_report = None

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# ==========================================
# 3. CACHED ENGINE LOADER
# ==========================================
@st.cache_resource
def get_prediction_engine():
    """Initializes and caches the prediction engine."""
    return HypoPredictionEngine()


engine = get_prediction_engine()


# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("### 🩸 **HypoGuardian AI**")
    st.caption("Hypoglycemia Early-Warning Prototype")
    st.markdown("---")

    selected_page = st.radio(
        "Navigation",
        [
            "🏠 Home / Dashboard",
            "📁 Upload Glucose Data",
            "📊 Glucose Analytics",
            "🔮 Early-Warning Prediction",
            "📜 Prediction History",
            "📈 Model Performance",
            "ℹ️ About the Project",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### **System Status**")
    engine_status = engine.get_status()
    if engine_status["ready"]:
        st.success(f"🟢 **Model Active**: {engine_status['model_name']} (v{engine_status['version']})")
    else:
        st.warning("⚠️ **Model Offline**: Artifacts missing")

    if st.session_state.current_df is not None:
        st.info(f"📁 **Active Dataset**: {len(st.session_state.current_df):,} readings loaded")
    else:
        st.caption("No data loaded in session")

    st.markdown("---")
    st.caption("Academic Research Prototype\nB.Tech Computer Science Project")


# Global Clinical Disclaimer
st.markdown(
    """
    <div class="disclaimer-banner">
        ⚠️ <b>Medical & Research Disclaimer:</b> HypoGuardian AI is an educational research prototype. 
        Its predictions are experimental and are not intended for diagnosis, treatment decisions, 
        insulin dosing, or emergency medical use. Do not rely on this application instead of an approved 
        Continuous Glucose Monitoring (CGM) device or professional medical guidance.
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# PAGE 1: HOME / DASHBOARD
# ==========================================
if selected_page == "🏠 Home / Dashboard":
    st.title("🩸 HypoGuardian AI: Dashboard")
    st.markdown(
        "**AI-Powered 30-Minute Hypoglycemia Early Warning System using Long Short-Term Memory (LSTM) Networks**"
    )

    # Top KPI Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    total_readings = len(st.session_state.current_df) if st.session_state.current_df is not None else 0
    latest_glucose = (
        float(st.session_state.current_df.iloc[-1]["glucose"])
        if st.session_state.current_df is not None and not st.session_state.current_df.empty
        else None
    )

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Uploaded Readings</div>
                <div class="metric-value">{total_readings:,}</div>
                <div class="metric-sub">{'Data loaded' if total_readings > 0 else 'No file uploaded'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        val_str = f"{latest_glucose:.1f} mg/dL" if latest_glucose is not None else "--"
        sub_str = "Normal" if (latest_glucose and latest_glucose >= 70) else ("Hypo Alert!" if latest_glucose else "No data")
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Latest Blood Glucose</div>
                <div class="metric-value">{val_str}</div>
                <div class="metric-sub">{sub_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        eligible = total_readings >= SEQUENCE_LENGTH
        elig_str = "Eligible (12+ steps)" if eligible else f"Need {SEQUENCE_LENGTH}+ steps"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Prediction Status</div>
                <div class="metric-value">{'Ready' if eligible and engine.is_ready() else 'Standby'}</div>
                <div class="metric-sub">{elig_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        last_pred = st.session_state.prediction_history[-1] if st.session_state.prediction_history else None
        pred_val = f"{last_pred['risk_level'].split(':')[0]}" if last_pred else "--"
        pred_sub = f"Prob: {last_pred['raw_probability']:.1%}" if last_pred else "No predictions made"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Latest Model Output</div>
                <div class="metric-value">{pred_val}</div>
                <div class="metric-sub">{pred_sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### **Continuous Glucose Profile**")
    if st.session_state.current_df is not None and not st.session_state.current_df.empty:
        fig = create_cgm_trend_chart(st.session_state.current_df)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### **Recent Readings**")
        st.dataframe(
            st.session_state.current_df.tail(10)[["patient_id", "timestamp", "glucose"]],
            use_container_width=True,
        )
    else:
        st.info(
            "👋 **Welcome!** To get started:\n\n"
            "1. Navigate to **📁 Upload Glucose Data** on the left sidebar.\n"
            "2. Download our pre-generated sample CSV or upload your own CGM file.\n"
            "3. Inspect historical trends on **📊 Glucose Analytics**.\n"
            "4. Run 30-minute early-warning inference on **🔮 Early-Warning Prediction**."
        )


# ==========================================
# PAGE 2: UPLOAD GLUCOSE DATA
# ==========================================
elif selected_page == "📁 Upload Glucose Data":
    st.title("📁 Upload & Validate Glucose Data")
    st.markdown("Upload continuous blood glucose readings (CSV) to inspect and run model inference.")

    # Schema specification & sample download
    with st.expander("ℹ️ **Supported CSV Schema & Specifications**", expanded=True):
        st.markdown(
            """
            Your CSV must contain at least two columns:
            - `timestamp`: Timestamp of reading (e.g., `YYYY-MM-DD HH:MM:SS` or `DD-MM-YYYY HH:MM:SS`)
            - `glucose`: Blood glucose measurement in **mg/dL** (typical range: 40 - 400 mg/dL)
            - `patient_id` *(optional)*: Patient identifier string (e.g., `P01`)
            """
        )

        sample_path = DATA_SAMPLE_DIR / "synthetic_cgm_sample.csv"
        if sample_path.exists():
            with open(sample_path, "r", encoding="utf-8") as f:
                sample_csv_bytes = f.read()
            st.download_button(
                label="📥 Download Synthetic Demonstration CSV Sample",
                data=sample_csv_bytes,
                file_name="hypoguardian_sample_cgm.csv",
                mime="text/csv",
                help="Click to download a realistic 24-hour synthetic CGM test file.",
            )

    uploaded_file = st.file_uploader("Upload Blood Glucose CSV File", type=["csv"])

    # Load Demo Data Button if no file is uploaded
    if uploaded_file is None and st.session_state.current_df is None:
        if st.button("🚀 Load Pre-Packaged Demo Sample"):
            sample_path = DATA_SAMPLE_DIR / "synthetic_cgm_sample.csv"
            raw_df = load_glucose_csv(sample_path)
            clean_df, audit = preprocess_glucose_data(raw_df)
            st.session_state.current_df = clean_df
            st.session_state.audit_report = audit
            st.success("Loaded pre-packaged demo dataset successfully!")
            st.rerun()

    if uploaded_file is not None:
        try:
            raw_df = load_glucose_csv(uploaded_file)
            clean_df, audit = preprocess_glucose_data(raw_df)
            st.session_state.current_df = clean_df
            st.session_state.audit_report = audit
            st.success(f"Successfully processed {len(clean_df):,} valid glucose readings!")
        except Exception as e:
            st.error(f"Error parsing uploaded file: {str(e)}")

    # Display audit report if data is active
    if st.session_state.audit_report is not None:
        audit = st.session_state.audit_report
        st.markdown("### **Data Cleaning & Preprocessing Audit**")

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Initial Rows", f"{audit.initial_rows:,}")
        a2.metric("Clean Rows Retained", f"{audit.clean_rows:,}")
        a3.metric("Duplicates Removed", f"{audit.duplicates_removed}")
        a4.metric("Sampling Gaps (>15m)", f"{audit.sampling_gaps_count}")

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Missing Timestamps", f"{audit.missing_timestamps}")
        b2.metric("Missing Glucose", f"{audit.missing_glucose}")
        b3.metric("Out of Bounds (<20 or >500)", f"{audit.out_of_bounds_glucose}")
        b4.metric("Median Interval", f"{audit.median_interval_minutes} min" if audit.median_interval_minutes else "N/A")

        st.markdown("#### **Preview of Cleaned Data**")
        st.dataframe(st.session_state.current_df.head(15), use_container_width=True)


# ==========================================
# PAGE 3: GLUCOSE ANALYTICS
# ==========================================
elif selected_page == "📊 Glucose Analytics":
    st.title("📊 Continuous Glucose Monitoring Analytics")

    if st.session_state.current_df is None or st.session_state.current_df.empty:
        st.warning("Please upload a CSV file or load demo data in 'Upload Glucose Data' first.")
    else:
        df = st.session_state.current_df

        # Patient Filter if multiple patients exist
        patients = df["patient_id"].unique().tolist()
        selected_patient = st.selectbox("Select Patient Identifier", patients) if len(patients) > 1 else patients[0]
        patient_df = df[df["patient_id"] == selected_patient].copy()

        # Clinical Summary Statistics
        st.markdown("### **Glycemic Statistics**")
        s1, s2, s3, s4, s5 = st.columns(5)
        g_min = float(patient_df["glucose"].min())
        g_max = float(patient_df["glucose"].max())
        g_mean = float(patient_df["glucose"].mean())
        g_median = float(patient_df["glucose"].median())
        g_std = float(patient_df["glucose"].std())

        s1.metric("Minimum Glucose", f"{g_min:.1f} mg/dL")
        s2.metric("Maximum Glucose", f"{g_max:.1f} mg/dL")
        s3.metric("Mean Glucose", f"{g_mean:.1f} mg/dL")
        s4.metric("Median Glucose", f"{g_median:.1f} mg/dL")
        s5.metric("Std Deviation", f"{g_std:.1f} mg/dL")

        # CGM Profile Chart
        st.markdown("### **Interactive Continuous Glucose Profile**")
        trend_fig = create_cgm_trend_chart(patient_df, patient_id=selected_patient)
        st.plotly_chart(trend_fig, use_container_width=True)

        # Time In Range & Distribution
        col_pie, col_hist = st.columns(2)
        with col_pie:
            st.markdown("#### **Time-in-Range (TIR) Breakdown**")
            pie_fig = create_time_in_range_pie(patient_df)
            st.plotly_chart(pie_fig, use_container_width=True)

        with col_hist:
            st.markdown("#### **Glucose Frequency Distribution**")
            hist_fig = px.histogram(
                patient_df,
                x="glucose",
                nbins=30,
                color_discrete_sequence=["#0284c7"],
                labels={"glucose": "Glucose (mg/dL)"},
            )
            hist_fig.add_vline(x=70, line_dash="dash", line_color="#ef4444", annotation_text="Hypo: 70")
            hist_fig.add_vline(x=180, line_dash="dash", line_color="#f59e0b", annotation_text="Hyper: 180")
            hist_fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(hist_fig, use_container_width=True)


# ==========================================
# PAGE 4: EARLY-WARNING PREDICTION
# ==========================================
elif selected_page == "🔮 Early-Warning Prediction":
    st.title("🔮 30-Minute Hypoglycemia Early Warning")
    st.markdown(
        "Evaluates the preceding **1 hour of continuous readings** (12 steps $\\approx$ 55 minutes) "
        "and estimates the probability of blood glucose dropping below **70 mg/dL** in the next **30 minutes**."
    )

    if not engine.is_ready():
        st.error(
            "⚠️ **Model Artifacts Not Loaded**: Trained model or scaler was not found. "
            "Please run `python -m src.train_model` in the terminal to train the LSTM network."
        )
    elif st.session_state.current_df is None or len(st.session_state.current_df) < SEQUENCE_LENGTH:
        st.warning(
            f"Please upload data containing at least {SEQUENCE_LENGTH} consecutive 5-minute readings "
            "to form an eligible input sequence."
        )
    else:
        df = st.session_state.current_df
        st.markdown("### **Select Historical Input Window**")

        # Allow user to pick prediction point (latest or custom)
        pred_mode = st.radio("Window Selection", ["Use Latest 12 Readings (~55 mins)", "Select Historical Timestamp Window"], horizontal=True)

        if pred_mode == "Use Latest 12 Readings (~55 mins)":
            window_df = df.iloc[-SEQUENCE_LENGTH:].copy()
        else:
            eligible_indices = list(range(SEQUENCE_LENGTH, len(df) + 1))
            selected_idx = st.slider("Select Window End Index", min_value=SEQUENCE_LENGTH, max_value=len(df), value=len(df))
            window_df = df.iloc[selected_idx - SEQUENCE_LENGTH : selected_idx].copy()

        st.markdown(
            f"**Selected Sequence:** `{window_df.iloc[0]['timestamp']}` to `{window_df.iloc[-1]['timestamp']}` "
            f"({len(window_df)} readings)"
        )

        # Classification threshold configuration
        col_thresh, col_btn = st.columns([2, 1])
        with col_thresh:
            threshold = st.slider(
                "Decision Threshold (Sensitivity Tuning)",
                min_value=0.10,
                max_value=0.90,
                value=float(engine.decision_threshold),
                step=0.05,
                help="Lowering the threshold increases sensitivity (catches more true hypos but increases false alarms).",
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            run_pred = st.button("🚀 Run HypoGuardian AI Prediction", type="primary", use_container_width=True)

        if run_pred:
            try:
                result = engine.predict(window_df, override_threshold=threshold)

                # Store in session history
                st.session_state.prediction_history.append(result)

                st.markdown("---")
                st.markdown("### **Inference Output**")

                res_col1, res_col2 = st.columns([1, 1])

                with res_col1:
                    if result["is_hypo_predicted"]:
                        st.markdown(
                            f"""
                            <div class="risk-badge-high" style="font-size: 1.1rem; padding: 10px 20px;">
                                🚨 <b>HIGH RISK: Hypoglycemia Predicted within 30 min</b>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="risk-badge-low" style="font-size: 1.1rem; padding: 10px 20px;">
                                ✅ <b>LOW RISK: Glycemia Expected to Remain Stable</b>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown(f"**Clinical Recommendation:** {result['clinical_advice']}")

                    st.markdown("#### **Model Inference Details**")
                    st.write(f"- **Uncalibrated Model Probability:** `{result['raw_probability']:.1%}`")
                    st.write(f"- **Active Classification Threshold:** `{result['threshold']:.1%}`")
                    st.write(f"- **Confidence Score:** `{result['confidence_score']:.1f}%`")
                    st.write(f"- **Prediction Timestamp (t_0):** `{result['prediction_timestamp']}`")
                    st.write(f"- **Model Version:** `{result['model_version']}`")

                with res_col2:
                    pred_fig = create_prediction_window_chart(
                        input_df=window_df,
                        predicted_prob=result["raw_probability"],
                        threshold=result["threshold"],
                        is_hypo_predicted=result["is_hypo_predicted"],
                    )
                    st.plotly_chart(pred_fig, use_container_width=True)

            except InvalidSequenceError as e:
                st.error(f"❌ Input validation failed: {str(e)}")
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")


# ==========================================
# PAGE 5: PREDICTION HISTORY
# ==========================================
elif selected_page == "📜 Prediction History":
    st.title("📜 Session Prediction Audit History")
    st.markdown("Log of all early-warning predictions generated during this session.")

    if not st.session_state.prediction_history:
        st.info("No predictions recorded yet in this session. Run predictions in 'Early-Warning Prediction' to populate this log.")
    else:
        history_records = []
        for item in st.session_state.prediction_history:
            history_records.append({
                "Timestamp (t_0)": item["prediction_timestamp"],
                "Latest Glucose (mg/dL)": item["latest_glucose"],
                "Model Probability": f"{item['raw_probability']:.1%}",
                "Decision Threshold": f"{item['threshold']:.1%}",
                "Result": "🚨 HIGH RISK" if item["is_hypo_predicted"] else "✅ NORMAL",
                "Confidence": f"{item['confidence_score']:.1f}%",
                "Model Version": item["model_version"],
            })

        history_df = pd.DataFrame(history_records)
        st.dataframe(history_df, use_container_width=True)

        col_dl, col_clr = st.columns([2, 1])
        with col_dl:
            csv_bytes = history_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Session Prediction History (CSV)",
                data=csv_bytes,
                file_name="hypoguardian_prediction_history.csv",
                mime="text/csv",
            )
        with col_clr:
            if st.button("🗑️ Clear Session History"):
                st.session_state.prediction_history = []
                st.rerun()


# ==========================================
# PAGE 6: MODEL PERFORMANCE
# ==========================================
elif selected_page == "📈 Model Performance":
    st.title("📈 Model Evaluation & Performance Statistics")
    st.markdown("Performance benchmarks computed on the strictly held-out test split.")

    if not EVALUATION_RESULTS_PATH.exists():
        st.warning(
            "Evaluation results not found. Please run `python -m src.evaluate_model` to compute test metrics."
        )
    else:
        with open(EVALUATION_RESULTS_PATH, "r", encoding="utf-8") as f:
            eval_payload = json.load(f)

        m = eval_payload["metrics"]
        st.markdown(f"**Evaluation Partition:** Held-out test set ({eval_payload['test_samples_count']} sequences)")

        # Performance KPI cards
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy", f"{m['accuracy']:.1%}")
        k2.metric("Balanced Accuracy", f"{m['balanced_accuracy']:.1%}")
        k3.metric("Sensitivity (Recall)", f"{m['sensitivity_recall']:.1%}", help="Percentage of true hypoglycemia events anticipated.")
        k4.metric("Specificity", f"{m['specificity']:.1%}", help="Percentage of normal glycemic intervals correctly classified.")

        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Precision (PPV)", f"{m['precision']:.1%}")
        k6.metric("F1-Score", f"{m['f1_score']:.3f}")
        k7.metric("ROC-AUC", f"{m['roc_auc']:.3f}" if m['roc_auc'] else "N/A")
        k8.metric("PR-AUC", f"{m['pr_auc']:.3f}" if m['pr_auc'] else "N/A")

        # Confusion Matrix Heatmap
        st.markdown("### **Confusion Matrix & Alarm Distribution**")
        cm_col, loss_col = st.columns(2)

        with cm_col:
            cm_fig = create_confusion_matrix_chart(eval_payload["confusion_matrix"])
            st.plotly_chart(cm_fig, use_container_width=True)

        with loss_col:
            history_path = MODELS_DIR / "training_history.json"
            if history_path.exists():
                with open(history_path, "r", encoding="utf-8") as f:
                    hist_data = json.load(f)
                train_fig = create_training_curves_chart(hist_data)
                st.plotly_chart(train_fig, use_container_width=True)
            else:
                st.caption("Training history curve not available.")

        # Class Imbalance Analysis Note
        with st.expander("🎓 **Why Accuracy is Misleading in Healthcare & Hypoglycemia Prediction**", expanded=True):
            st.markdown(
                f"""
                - **Class Imbalance**: In this test cohort, true hypoglycemia occurs in only **{m['prevalence']:.1%}** of time windows.
                - **The Naive Predictor Fallacy**: A model that simply predicts 'No Hypoglycemia' 100% of the time would achieve **{1.0 - m['prevalence']:.1%} accuracy**, but would have a **0% sensitivity** (missing every single life-threatening event!).
                - **Clinical Metrics that Matter**:
                  - **Sensitivity (Recall = {m['sensitivity_recall']:.1%})**: How reliably the model alerts the patient to low glucose before neuroglycopenic symptoms occur.
                  - **Specificity ({m['specificity']:.1%})**: How effectively the model avoids alert fatigue (unnecessary alarms when glucose is safe).
                  - **ROC-AUC ({m['roc_auc']:.3f})**: Demonstrates the network's discriminative ability independent of the decision threshold.
                """
            )


# ==========================================
# PAGE 7: ABOUT THE PROJECT & VIVA PREPARATION
# ==========================================
elif selected_page == "ℹ️ About the Project":
    st.title("ℹ️ About HypoGuardian AI & Viva Preparation")

    st.markdown(
        """
        ### **1. Clinical Background & Motivation**
        In Type 1 and advanced Type 2 Diabetes, continuous glucose monitoring (CGM) provides real-time interstitial 
        glucose updates every 5 minutes. However, reactive alarms only ring after blood glucose has already crossed 
        below the critical $70\\text{ mg/dL}$ boundary. 
        
        **HypoGuardian AI** addresses this limitation by looking at the trajectory of the past hour (12 readings) 
        and alerting the patient **30 minutes in advance**, giving them adequate time to take 15 grams of fast-acting 
        carbohydrates (the clinical "Rule of 15").

        ---
        ### **2. Deep Learning Architecture**
        - **Model Type**: Long Short-Term Memory (LSTM) Recurrent Neural Network
        - **Input Dimensions**: `(Batch Size, 12 Time Steps, 1 Glucose Feature)`
        - **LSTM Layer**: 64 units extracting temporal trend dependencies
        - **Regularization**: Dropout (0.30 and 0.20) to prevent overfitting
        - **Dense Representation**: 32 units with ReLU non-linearity
        - **Classification Output**: 1 unit with Sigmoid activation ($0.0$ to $1.0$)
        - **Loss Function**: Binary Cross-Entropy with balanced class weighting

        ---
        ### **3. Time-Series Sequence Formulation**
        - **Sequence Length**: 12 readings $\\approx$ 55 minutes
        - **Sampling Interval**: 5 minutes
        - **Prediction Target ($y$)**: Binary ($0$ or $1$)
          $$y = 1 \\iff \\min(G_{t+1}, \\dots, G_{t+6}) < 70\\text{ mg/dL}$$
        - **Zero Leakage**: Scaler is fitted strictly on the training partition. 
          Sensor disconnect gaps ($> 15$ min) and patient boundaries are strictly isolated.

        ---
        ### **4. Technology Stack**
        - **Deep Learning**: TensorFlow 2.x / Keras
        - **Data Processing**: NumPy, Pandas, Scikit-Learn
        - **Web Application**: Streamlit
        - **Interactive Plotting**: Plotly Graph Objects & Express
        - **Testing**: Pytest
        """
    )
