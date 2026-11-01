"""HypoGuardian AI: Interactive Visualization Engine.

Generates medical-grade Plotly interactive charts for CGM profiles, prediction windows,
glycemic time-in-range distributions, confusion matrices, and ROC/PR curves.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import HYPO_THRESHOLD_MG_DL


def create_cgm_trend_chart(
    df: pd.DataFrame,
    patient_id: Optional[str] = None,
    title: str = "Continuous Glucose Monitor (CGM) Trend Analysis",
) -> go.Figure:
    """Renders interactive CGM time-series plot with clinical glycemic target zones.

    - Target Glycemic Range: 70 - 180 mg/dL (Green/Teal zone)
    - Hypoglycemia Risk Zone: < 70 mg/dL (Red/Warning zone)
    - Hyperglycemia Zone: > 180 mg/dL (Amber zone)
    """
    plot_df = df.copy()
    if patient_id and "patient_id" in plot_df.columns:
        plot_df = plot_df[plot_df["patient_id"] == patient_id]

    fig = go.Figure()

    # Hypoglycemia boundary shaded zone [0, 70]
    fig.add_hrect(
        y0=0,
        y1=HYPO_THRESHOLD_MG_DL,
        fillcolor="rgba(239, 68, 68, 0.12)",
        layer="below",
        line_width=0,
        annotation_text="Hypoglycemia Risk (< 70 mg/dL)",
        annotation_position="bottom right",
        annotation_font=dict(color="#ef4444", size=11),
    )

    # Target Glycemic Range [70, 180]
    fig.add_hrect(
        y0=HYPO_THRESHOLD_MG_DL,
        y1=180.0,
        fillcolor="rgba(16, 185, 129, 0.08)",
        layer="below",
        line_width=0,
        annotation_text="Target Glycemic Range (70-180 mg/dL)",
        annotation_position="top right",
        annotation_font=dict(color="#10b981", size=11),
    )

    # Main CGM Trace
    fig.add_trace(
        go.Scatter(
            x=plot_df["timestamp"],
            y=plot_df["glucose"],
            mode="lines+markers",
            name="Glucose (mg/dL)",
            line=dict(color="#0284c7", width=2.5),
            marker=dict(size=4, color="#0369a1"),
            hovertemplate="<b>Time:</b> %{x|%Y-%m-%d %H:%M}<br><b>Glucose:</b> %{y:.1f} mg/dL<extra></extra>",
        )
    )

    # Critical 70 mg/dL reference line
    fig.add_hline(
        y=HYPO_THRESHOLD_MG_DL,
        line_dash="dash",
        line_color="#dc2626",
        line_width=1.5,
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family="Inter, sans-serif")),
        xaxis=dict(
            title="Time",
            gridcolor="#f1f5f9",
            showgrid=True,
        ),
        yaxis=dict(
            title="Blood Glucose (mg/dL)",
            gridcolor="#f1f5f9",
            range=[30, max(240, plot_df["glucose"].max() + 20) if not plot_df.empty else 250],
            showgrid=True,
        ),
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_prediction_window_chart(
    input_df: pd.DataFrame,
    predicted_prob: float,
    threshold: float,
    is_hypo_predicted: bool,
) -> go.Figure:
    """Visualizes the 1-hour input sequence (12 readings) and projects the 30-min forecast horizon."""
    fig = go.Figure()

    # Hypoglycemia Zone
    fig.add_hrect(
        y0=0,
        y1=HYPO_THRESHOLD_MG_DL,
        fillcolor="rgba(239, 68, 68, 0.15)",
        layer="below",
        line_width=0,
        annotation_text="Hypoglycemia Alert Zone (< 70 mg/dL)",
        annotation_position="bottom right",
        annotation_font=dict(color="#dc2626", size=10),
    )

    # Historical 12 readings (Past 55 minutes)
    fig.add_trace(
        go.Scatter(
            x=input_df["timestamp"],
            y=input_df["glucose"],
            mode="lines+markers",
            name="Observed History (12 Steps)",
            line=dict(color="#2563eb", width=3),
            marker=dict(size=6, color="#1d4ed8"),
            hovertemplate="<b>Past Reading:</b> %{y:.1f} mg/dL<br>%{x|%H:%M}<extra></extra>",
        )
    )

    # Latest reading (t_0)
    last_row = input_df.iloc[-1]
    last_time = pd.to_datetime(last_row["timestamp"])
    last_glucose = float(last_row["glucose"])

    # 30-min horizon projection
    future_time = last_time + pd.Timedelta(minutes=30)

    # Prediction forecast line
    forecast_color = "#dc2626" if is_hypo_predicted else "#059669"
    label_text = "HIGH RISK: Hypo Expected" if is_hypo_predicted else "LOW RISK: Stable"

    fig.add_trace(
        go.Scatter(
            x=[last_time, future_time],
            y=[last_glucose, 60.0 if is_hypo_predicted else last_glucose],
            mode="lines",
            name=f"30-Min Forecast ({label_text})",
            line=dict(color=forecast_color, width=2.5, dash="dot"),
            hoverinfo="skip",
        )
    )

    fig.add_hline(
        y=HYPO_THRESHOLD_MG_DL,
        line_dash="dash",
        line_color="#ef4444",
        line_width=1.5,
    )

    fig.update_layout(
        title=dict(
            text=f"Sequence Forecast Window | Risk Score: {predicted_prob:.1%} (Threshold: {threshold:.0%})",
            font=dict(size=14),
        ),
        xaxis=dict(title="Time", showgrid=True, gridcolor="#f8fafc"),
        yaxis=dict(title="Glucose (mg/dL)", showgrid=True, gridcolor="#f8fafc"),
        template="plotly_white",
        margin=dict(l=30, r=30, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_time_in_range_pie(df: pd.DataFrame) -> go.Figure:
    """Computes and renders standard clinical Time-in-Range (TIR) metrics:

    - TBR (Time Below Range < 70 mg/dL)
    - TIR (Time In Range 70-180 mg/dL)
    - TAR (Time Above Range > 180 mg/dL)
    """
    total = len(df)
    if total == 0:
        return go.Figure()

    tbr_count = (df["glucose"] < 70.0).sum()
    tir_count = ((df["glucose"] >= 70.0) & (df["glucose"] <= 180.0)).sum()
    tar_count = (df["glucose"] > 180.0).sum()

    labels = ["Below Range (<70 mg/dL)", "Target Range (70-180 mg/dL)", "Above Range (>180 mg/dL)"]
    values = [tbr_count, tir_count, tar_count]
    colors = ["#ef4444", "#10b981", "#f59e0b"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
            )
        ]
    )
    fig.update_layout(
        title=dict(text="Clinical Time-In-Range (TIR) Distribution", font=dict(size=15)),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
    )
    return fig


def create_confusion_matrix_chart(cm_data: List[List[int]]) -> go.Figure:
    """Renders annotated confusion matrix heatmap."""
    labels = ["Normal (0)", "Hypo Event (1)"]
    z = cm_data

    annotations = []
    cell_names = [["True Negatives (TN)", "False Positives (FP)"],
                  ["False Negatives (FN)", "True Positives (TP)"]]

    for i in range(2):
        for j in range(2):
            val = z[i][j]
            name = cell_names[i][j]
            annotations.append(
                dict(
                    x=labels[j],
                    y=labels[i],
                    text=f"<b>{name}</b><br><span style='font-size:18px;'>{val}</span>",
                    showarrow=False,
                    font=dict(color="white" if val > np.max(z) * 0.5 else "#0f172a", size=13),
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=False,
        )
    )
    fig.update_layout(
        title=dict(text="Confusion Matrix (Held-out Test Set)", font=dict(size=15)),
        xaxis=dict(title="Model Prediction", side="bottom"),
        yaxis=dict(title="True Clinical Observation", autorange="reversed"),
        annotations=annotations,
        template="plotly_white",
        margin=dict(l=60, r=60, t=50, b=60),
    )
    return fig


def create_training_curves_chart(history_dict: dict) -> go.Figure:
    """Plots training and validation loss & AUC across epochs."""
    epochs = list(range(1, len(history_dict.get("loss", [])) + 1))
    fig = go.Figure()

    if "loss" in history_dict:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history_dict["loss"],
                mode="lines",
                name="Train Loss",
                line=dict(color="#3b82f6", width=2),
            )
        )
    if "val_loss" in history_dict:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history_dict["val_loss"],
                mode="lines",
                name="Validation Loss",
                line=dict(color="#ef4444", width=2, dash="dash"),
            )
        )

    fig.update_layout(
        title=dict(text="LSTM Training vs. Validation Loss Across Epochs", font=dict(size=14)),
        xaxis=dict(title="Epoch", showgrid=True),
        yaxis=dict(title="Binary Cross-Entropy Loss", showgrid=True),
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
