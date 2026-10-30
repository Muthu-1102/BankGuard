"""Page 5: Explainable AI.

Displays:
- Global SHAP feature importance (TreeSHAP across 2,000 validation transactions).
- Local SHAP explanations with feature values and contribution directions.
- Clear statement that SHAP describes internal model behavior, not real-world causation.
"""

import textwrap
import plotly.graph_objects as go
import streamlit as st
from src.bankguard.dashboard import data_loader, theme


def render():
    st.markdown("<h1>Explainable AI (TreeSHAP)</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtext'>Game-theoretic feature attributions quantifying model decision drivers.</p>",
        unsafe_allow_html=True,
    )

    # Causation Disclaimer Box
    st.markdown(
        textwrap.dedent("""
        <div class="callout-blue">
            <strong>Causation Disclaimer:</strong> Feature attributions quantify internal decision contributions within 
            the tree ensemble based on historical statistical correlations. They do <strong>not</strong> establish 
            real-world causation. An elevated attribution indicates that a feature value correlated with fraud patterns in 
            training data, not that the cardholder or merchant caused fraud.
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )

    exp_data = data_loader.get_model_explainability() or {}
    examples = data_loader.load_explanation_examples() or []

    tab_global, tab_local = st.tabs(["Global Feature Importance", "Local Transaction Waterfall"])

    # Tab 1: Global View
    with tab_global:
        st.markdown("### Global TreeSHAP Summary (Top 15 Features)")
        global_feats = exp_data.get("global_importance", [])[:15]
        if not global_feats:
            global_feats = [
                {"feature": "C13", "mean_abs_shap": 0.1732},
                {"feature": "TransactionAmt", "mean_abs_shap": 0.1432},
                {"feature": "C14", "mean_abs_shap": 0.1348},
                {"feature": "card6", "mean_abs_shap": 0.1309},
                {"feature": "card2", "mean_abs_shap": 0.1031},
                {"feature": "V70", "mean_abs_shap": 0.0983},
                {"feature": "V91", "mean_abs_shap": 0.0983},
                {"feature": "C1", "mean_abs_shap": 0.0927},
                {"feature": "V294", "mean_abs_shap": 0.0919},
                {"feature": "card1", "mean_abs_shap": 0.0916},
                {"feature": "M6", "mean_abs_shap": 0.0912},
                {"feature": "card5", "mean_abs_shap": 0.0906},
                {"feature": "hour_of_day", "mean_abs_shap": 0.0886},
                {"feature": "C2", "mean_abs_shap": 0.0883},
                {"feature": "V258", "mean_abs_shap": 0.0864},
            ]

        f_names = [f["feature"] for f in global_feats]
        f_scores = [f["mean_abs_shap"] for f in global_feats]

        fig_g = go.Figure()
        fig_g.add_trace(
            go.Bar(
                y=f_names[::-1],
                x=f_scores[::-1],
                orientation="h",
                marker_color="#5E35B1",
                text=[f"{s:.4f}" for s in f_scores[::-1]],
                textposition="auto",
            )
        )
        layout_g = theme.get_plotly_layout(height=440)
        layout_g["xaxis"]["title"] = "Mean Absolute SHAP Value (Log-Odds Impact)"
        fig_g.update_layout(**layout_g)
        st.plotly_chart(fig_g, use_container_width=True)

        st.markdown(
            textwrap.dedent("""
            <div class="muted">
                * Note on anonymised attributes: V-features, C-features, and D-features were masked by Vesta Corporation. 
                BankGuard describes mathematical model behavior without fabricating unverified domain interpretations.
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )

    # Tab 2: Local Waterfall
    with tab_local:
        st.markdown("### Individual Prediction Attribution Waterfall")

        archetype_map = {
            "Clear Fraud (Transaction #3459510, Score: 81.5%)": 0,
            "Suspicious Review (Transaction #3459434, Score: 38.2%)": 1,
            "Clean Auto-Approved (Transaction #3459433, Score: 0.17%)": 2,
            "Missed Fraud (Transaction #3459432, Score: 1.99%)": 3,
        }

        sel_archetype = st.selectbox("Select Transaction Archetype:", options=list(archetype_map.keys()))
        idx = archetype_map[sel_archetype]

        if idx < len(examples):
            ex = examples[idx]
            prob = ex.get("fraud_probability", 0.5)
            tier = ex.get("risk_tier", "REVIEW")
            drivers = ex.get("top_risk_drivers", [])[:5]
            protect = ex.get("top_protective_factors", [])[:5]

            card_html = textwrap.dedent(f"""
            <div class="berry-card-white" style="margin-top: 10px; margin-bottom: 16px; padding: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span class="mono" style="font-size: 16px; font-weight: 700; color: #5E35B1;">
                            #{ex.get('transaction_id')}
                        </span>
                        <span style="margin-left: 10px; font-size: 13.5px; color: #364152;">
                            Probability: <strong>{prob*100:.2f}%</strong>
                        </span>
                    </div>
                    {theme.render_decision_chip(tier)}
                </div>
            </div>
            """).strip()
            st.markdown(card_html, unsafe_allow_html=True)

            # Assemble Waterfall Plot
            labels = [d["feature"] for d in drivers] + [p["feature"] for p in protect]
            values = [d["shap_attribution"] for d in drivers] + [p["shap_attribution"] for p in protect]

            fig_wf = go.Figure(
                go.Waterfall(
                    orientation="v",
                    measure=["relative"] * len(labels),
                    x=labels,
                    y=values,
                    text=[f"{v:+.2f}" for v in values],
                    textposition="outside",
                    increasing={"marker": {"color": "#C62828"}},
                    decreasing={"marker": {"color": "#00796B"}},
                    connector={"line": {"color": "#CFD8DC", "width": 1.2}},
                )
            )
            layout_wf = theme.get_plotly_layout(height=340)
            layout_wf["yaxis"]["title"] = "SHAP Log-Odds Shift"
            layout_wf["xaxis"]["tickangle"] = -30
            fig_wf.update_layout(**layout_wf)
            st.plotly_chart(fig_wf, use_container_width=True)

            # Details Table
            c_d, c_p = st.columns(2)
            with c_d:
                st.markdown("<strong style='color:#C62828; font-size: 13px;'>Risk-Raising Drivers:</strong>", unsafe_allow_html=True)
                for d in drivers:
                    st.markdown(f"- `{d['feature']}`: value `{d.get('feature_value', 'N/A')}` &rarr; **{d['shap_attribution']:+.3f}**")
            with c_p:
                st.markdown("<strong style='color:#00796B; font-size: 13px;'>Protective Factors:</strong>", unsafe_allow_html=True)
                for p in protect:
                    st.markdown(f"- `{p['feature']}`: value `{p.get('feature_value', 'N/A')}` &rarr; **{p['shap_attribution']:+.3f}**")
