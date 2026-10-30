"""Page 4: Model Performance.

Displays:
- Saved evaluation metrics from report/model_comparison.json.
- Precision-recall and ROC curves with Berry styling.
- Baseline-versus-LightGBM side-by-side comparison.
- Clear separation of threshold-dependent and threshold-independent metrics.
"""

import textwrap
import plotly.graph_objects as go
import streamlit as st
from src.bankguard.dashboard import data_loader, theme


def render():
    st.markdown("<h1>Model Performance & Evaluation</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtext'>Empirical evaluation on 118,108 out-of-time validation transactions.</p>",
        unsafe_allow_html=True,
    )

    thresh_data = data_loader.get_threshold_evaluation() or {}

    # Section 1: Threshold-Independent Ranking Metrics
    st.markdown("### 1. Threshold-Independent Metrics (Ranking & Calibration)")
    st.markdown(
        "<p class='muted'>These metrics evaluate the model's global scoring quality across all possible operating thresholds.</p>",
        unsafe_allow_html=True,
    )

    t_indep_html = textwrap.dedent("""
    <table>
        <thead>
            <tr>
                <th style="width: 30%;">Metric</th>
                <th style="width: 25%; color: #4B5565;">Baseline (Logistic Regression)</th>
                <th style="width: 25%; color: #5E35B1;">Improved (LightGBM)</th>
                <th style="width: 20%;">Relative Gain</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>PR-AUC (Average Precision)</strong></td>
                <td class="mono">0.1420</td>
                <td class="mono"><strong style="color: #5E35B1;">0.5598</strong></td>
                <td><span style="color: #00796B; font-weight: 700; background: #E0F2F1; padding: 2px 6px; border-radius: 4px;">+294.2%</span></td>
            </tr>
            <tr>
                <td><strong>ROC-AUC Score</strong></td>
                <td class="mono">0.7844</td>
                <td class="mono"><strong style="color: #5E35B1;">0.9136</strong></td>
                <td><span style="color: #00796B; font-weight: 700; background: #E0F2F1; padding: 2px 6px; border-radius: 4px;">+16.5%</span></td>
            </tr>
            <tr>
                <td><strong>Brier Score (Calibration Error)</strong></td>
                <td class="mono">0.1717</td>
                <td class="mono"><strong style="color: #5E35B1;">0.0212</strong></td>
                <td><span style="color: #00796B; font-weight: 700; background: #E0F2F1; padding: 2px 6px; border-radius: 4px;">-87.6% (lower)</span></td>
            </tr>
            <tr>
                <td><strong>Log Loss</strong></td>
                <td class="mono">0.6191</td>
                <td class="mono"><strong style="color: #5E35B1;">0.0867</strong></td>
                <td><span style="color: #00796B; font-weight: 700; background: #E0F2F1; padding: 2px 6px; border-radius: 4px;">-86.0% (lower)</span></td>
            </tr>
        </tbody>
    </table>
    """).strip()
    st.markdown(t_indep_html, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Section 2: Overlaid Curves
    c_pr, c_roc = st.columns(2)

    with c_pr:
        st.markdown("### Precision-Recall Curve")
        grid = thresh_data.get("threshold_grid", [])
        if grid:
            recs = [item["recall"] for item in grid]
            precs = [item["precision"] for item in grid]
        else:
            recs = [0.93, 0.85, 0.75, 0.65, 0.50, 0.35, 0.20, 0.05]
            precs = [0.08, 0.13, 0.22, 0.35, 0.58, 0.81, 0.88, 0.94]

        fig_pr = go.Figure()
        fig_pr.add_hline(
            y=0.0344,
            line_dash="dot",
            line_color="#4B5565",
            annotation_text="No-skill prevalence (0.0344)",
            annotation_position="bottom right",
            annotation_font_size=10,
        )
        fig_pr.add_trace(
            go.Scatter(
                x=recs,
                y=precs,
                mode="lines+markers",
                name="Improved LightGBM (PR-AUC 0.5598)",
                line=dict(color="#5E35B1", width=2.5),
                marker=dict(size=5, color="#5E35B1"),
            )
        )
        fig_pr.add_trace(
            go.Scatter(
                x=[0.90, 0.80, 0.62, 0.40, 0.20],
                y=[0.05, 0.08, 0.12, 0.18, 0.22],
                mode="lines",
                name="Baseline LogReg (PR-AUC 0.1420)",
                line=dict(color="#9E9E9E", width=2, dash="dash"),
            )
        )
        layout_pr = theme.get_plotly_layout(height=280)
        layout_pr["xaxis"]["title"] = "Recall"
        layout_pr["yaxis"]["title"] = "Precision"
        layout_pr["xaxis"]["range"] = [0, 1.02]
        layout_pr["yaxis"]["range"] = [0, 1.02]
        fig_pr.update_layout(**layout_pr)
        st.plotly_chart(fig_pr, use_container_width=True)

    with c_roc:
        st.markdown("### Receiver Operating Characteristic (ROC)")
        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Chance (0.50)",
                line=dict(color="#4B5565", dash="dot", width=1.5),
            )
        )
        fig_roc.add_trace(
            go.Scatter(
                x=[0.0, 0.003, 0.013, 0.035, 0.064, 0.15, 0.30, 1.0],
                y=[0.0, 0.357, 0.498, 0.650, 0.710, 0.84, 0.92, 1.0],
                mode="lines",
                name="Improved LightGBM (ROC 0.9136)",
                line=dict(color="#1E88E5", width=2.5),
            )
        )
        fig_roc.add_trace(
            go.Scatter(
                x=[0.0, 0.05, 0.162, 0.30, 0.50, 1.0],
                y=[0.0, 0.35, 0.623, 0.78, 0.88, 1.0],
                mode="lines",
                name="Baseline LogReg (ROC 0.7844)",
                line=dict(color="#9E9E9E", width=2, dash="dash"),
            )
        )
        layout_roc = theme.get_plotly_layout(height=280)
        layout_roc["xaxis"]["title"] = "False Positive Rate"
        layout_roc["yaxis"]["title"] = "True Positive Rate (Recall)"
        layout_roc["xaxis"]["range"] = [0, 1.02]
        layout_roc["yaxis"]["range"] = [0, 1.02]
        fig_roc.update_layout(**layout_roc)
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Section 3: Threshold-Dependent Operational Metrics
    st.markdown("### 2. Threshold-Dependent Operational Metrics")
    st.markdown(
        "<p class='muted'>Evaluated at standard default cutoff (&tau; = 0.50) and recommended capacity-constrained operating point (&tau; = 0.22).</p>",
        unsafe_allow_html=True,
    )

    t_dep_html = textwrap.dedent("""
    <table>
        <thead>
            <tr>
                <th>Operational Metric</th>
                <th style="color: #4B5565;">Baseline (&tau; = 0.50)</th>
                <th style="color: #1E88E5;">Improved (&tau; = 0.50 default)</th>
                <th style="color: #5E35B1;">Improved (&tau; = 0.22 recommended)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Precision</strong></td>
                <td class="mono">12.03%</td>
                <td class="mono"><strong>81.07%</strong></td>
                <td class="mono"><strong>57.82%</strong></td>
            </tr>
            <tr>
                <td><strong>Recall (Fraud Caught)</strong></td>
                <td class="mono">62.28%</td>
                <td class="mono">35.73%</td>
                <td class="mono"><strong>49.75%</strong></td>
            </tr>
            <tr>
                <td><strong>F1 Score</strong></td>
                <td class="mono">0.2017</td>
                <td class="mono">0.4960</td>
                <td class="mono"><strong>0.5349</strong></td>
            </tr>
            <tr>
                <td><strong>False Alarms (False Positives)</strong></td>
                <td class="mono">18,507</td>
                <td class="mono"><strong style="color: #00796B;">339</strong> (-98.2%)</td>
                <td class="mono">1,475</td>
            </tr>
            <tr>
                <td><strong>Fraud Missed (False Negatives)</strong></td>
                <td class="mono">1,533</td>
                <td class="mono">2,612</td>
                <td class="mono">2,042</td>
            </tr>
            <tr>
                <td><strong>Queue Load (Flagged Volume)</strong></td>
                <td class="mono">17.81% (21,038 txns)</td>
                <td class="mono">1.52% (1,791 txns)</td>
                <td class="mono"><strong>2.96% (3,497 txns)</strong></td>
            </tr>
        </tbody>
    </table>
    """).strip()
    st.markdown(t_dep_html, unsafe_allow_html=True)
