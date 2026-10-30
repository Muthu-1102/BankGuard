"""Page 3: Risk Analysis.

Displays:
- Interactive threshold selection (snapping to saved offline grid).
- Precision, recall, F1, false positives, false negatives, and review volume.
- Confusion matrix cards in Berry style.
- Clearly documented operational objectives and cost assumptions.
"""

import textwrap
import plotly.graph_objects as go
import streamlit as st
from src.bankguard.dashboard import data_loader, theme


def render():
    st.markdown("<h1>Risk Analysis & Policy Calibration</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtext'>Interactive threshold selection, trade-off optimization, and confusion matrix calibration.</p>",
        unsafe_allow_html=True,
    )

    thresh_data = data_loader.get_threshold_evaluation() or {}
    grid = thresh_data.get("threshold_grid", [])

    # Threshold Slider
    st.markdown("### Interactive Threshold Simulator")
    selected_tau = st.slider(
        "Operating Threshold (\u03c4):",
        min_value=0.01,
        max_value=0.90,
        value=0.22,
        step=0.01,
        help="Decision boundary: transactions with P(isFraud \u2265 \u03c4) are flagged for action.",
    )

    # Snap to nearest saved grid point
    if grid:
        item = min(grid, key=lambda x: abs(x["threshold"] - selected_tau))
    else:
        item = {
            "threshold": 0.22,
            "precision": 0.5782,
            "recall": 0.4975,
            "f1_score": 0.5349,
            "tp": 2022,
            "fp": 1475,
            "fn": 2042,
            "tn": 112569,
            "flagged_count": 3497,
            "flagged_rate_pct": 2.96,
            "total_cost_usd": 341270,
        }

    st.markdown(
        f"<div class='muted' style='margin-top:-6px; margin-bottom:14px;'>Snapped to nearest evaluated point: <strong class='mono' style='color:#5E35B1;'>\u03c4 = {item['threshold']:.2f}</strong></div>",
        unsafe_allow_html=True,
    )

    # 4 KPI Cards for the chosen threshold
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        theme.render_kpi_card(
            title="Precision",
            value=f"{item['precision'] * 100:.1f}%",
            caption="Share of flagged txns that are fraud",
            meaning=f"{int(item['fp']):,} false alarms generated.",
            accent_color="#5E35B1",
        )
    with c2:
        theme.render_kpi_card(
            title="Recall (Fraud Caught)",
            value=f"{item['recall'] * 100:.1f}%",
            caption="Share of real fraud detected",
            meaning=f"{int(item['tp']):,} of 4,064 fraud cases captured.",
            accent_color="#00796B",
        )
    with c3:
        theme.render_kpi_card(
            title="F1 Score",
            value=f"{item['f1_score']:.3f}",
            caption="Harmonic mean of precision & recall",
            meaning="Optimal balance point between friction and loss.",
            accent_color="#1E88E5",
        )
    with c4:
        theme.render_kpi_card(
            title="Review Queue Load",
            value=f"{item['flagged_rate_pct']:.2f}%",
            caption=f"{int(item['flagged_count']):,} transactions flagged",
            meaning="Workload dispatched to investigation team.",
            accent_color="#C0392B",
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Confusion Matrix (2x2 Cards)
    st.markdown("### Confusion Matrix at Chosen Threshold")
    tp = int(item["tp"])
    fp = int(item["fp"])
    fn = int(item["fn"])
    tn = int(item["tn"])

    cm_col1, cm_col2 = st.columns(2)
    with cm_col1:
        st.markdown(
            textwrap.dedent(f"""
            <div class="berry-card-white" style="border-left: 4px solid #00796B !important; margin-bottom: 12px; padding: 16px;">
                <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #4B5565;">Fraud Caught (True Positives)</div>
                <div style="font-size: 26px; font-weight: 800; color: #00796B; font-family: 'JetBrains Mono', monospace;">{tp:,}</div>
                <div class="muted">{(tp / 4064 * 100):.1f}% of actual fraud successfully intercepted</div>
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )
        st.markdown(
            textwrap.dedent(f"""
            <div class="berry-card-white" style="border-left: 4px solid #C62828 !important; padding: 16px;">
                <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #4B5565;">Fraud Missed (False Negatives)</div>
                <div style="font-size: 26px; font-weight: 800; color: #C62828; font-family: 'JetBrains Mono', monospace;">{fn:,}</div>
                <div class="muted">{(fn / 4064 * 100):.1f}% of fraud escaped undetected below cutoff</div>
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )

    with cm_col2:
        st.markdown(
            textwrap.dedent(f"""
            <div class="berry-card-white" style="border-left: 4px solid #1E88E5 !important; margin-bottom: 12px; padding: 16px;">
                <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #4B5565;">Legitimate Cleared (True Negatives)</div>
                <div style="font-size: 26px; font-weight: 800; color: #1E88E5; font-family: 'JetBrains Mono', monospace;">{tn:,}</div>
                <div class="muted">{(tn / 114044 * 100):.2f}% of honest transactions passed without delay</div>
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )
        st.markdown(
            textwrap.dedent(f"""
            <div class="berry-card-white" style="border-left: 4px solid #C0392B !important; padding: 16px;">
                <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: #4B5565;">False Alarms (False Positives)</div>
                <div style="font-size: 26px; font-weight: 800; color: #C0392B; font-family: 'JetBrains Mono', monospace;">{fp:,}</div>
                <div class="muted">{(fp / 114044 * 100):.2f}% of honest transactions incorrectly flagged</div>
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Operational Objectives & Cost Assumptions
    st.markdown("### Operational Objective Calibration")
    st.markdown(
        "<p class='subtext'>Pre-calculated operating thresholds based on specific business objectives and cost models.</p>",
        unsafe_allow_html=True,
    )

    opt_tbl_html = textwrap.dedent("""
    <table>
        <thead>
            <tr>
                <th>Objective Strategy</th>
                <th>Threshold (&tau;)</th>
                <th>Recall (Fraud Caught)</th>
                <th>Precision</th>
                <th>Review Queue</th>
                <th>Cost Model Assumption</th>
            </tr>
        </thead>
        <tbody>
            <tr style="background: #EDE7F6; border-left: 3px solid #5E35B1;">
                <td><strong>Capacity-Constrained Review</strong> <span style="background: #5E35B1; color: #FFFFFF; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: 700;">RECOMMENDED</span></td>
                <td class="mono"><strong>0.22</strong></td>
                <td><strong>49.75%</strong></td>
                <td><strong>57.82%</strong></td>
                <td class="mono">2.96% (3,497 txns)</td>
                <td>Subject to review capacity &le; 3.0% of daily volume</td>
            </tr>
            <tr>
                <td><strong>Max F1-Score</strong></td>
                <td class="mono">0.25</td>
                <td>47.61%</td>
                <td>62.26%</td>
                <td class="mono">2.63% (3,108 txns)</td>
                <td>argmax(F1-score) balancing precision and recall</td>
            </tr>
            <tr>
                <td><strong>Target Recall Guarantee (&ge;70%)</strong></td>
                <td class="mono">0.06</td>
                <td>71.01%</td>
                <td>28.41%</td>
                <td class="mono">8.60% (10,159 txns)</td>
                <td>Guarantees &ge;70% fraud detection rate</td>
            </tr>
            <tr>
                <td><strong>Minimum Total Operational Cost</strong></td>
                <td class="mono">0.06</td>
                <td>71.01%</td>
                <td>28.41%</td>
                <td class="mono">8.60% (10,159 txns)</td>
                <td>Loss per FN: $149 (avg amount) + Review cost: $10/txn</td>
            </tr>
        </tbody>
    </table>
    """).strip()
    st.markdown(opt_tbl_html, unsafe_allow_html=True)
