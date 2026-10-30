"""Page 2: Transaction Explorer.

Displays:
- Search and filters backed strictly by actual prediction data (artifacts/improved_predictions.npz).
- Transaction ID, fraud score, policy tier, and actual label.
- Detailed inspector view for the selected transaction.
- Zero fabricated transactions or labels.
"""

import textwrap
import plotly.graph_objects as go
import streamlit as st
from src.bankguard.dashboard import data_loader, theme


def render():
    st.markdown("<h1>Transaction Explorer</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtext'>Interactive inspection of real out-of-time validation transactions and policy decisions.</p>",
        unsafe_allow_html=True,
    )

    df_val = data_loader.load_validation_predictions(max_rows=5000)
    examples = data_loader.load_explanation_examples() or []

    if df_val is None or df_val.empty:
        st.markdown(
            textwrap.dedent("""
            <div class="callout-amber">
                <strong>Data Notice:</strong> Validation predictions artifact (<code>artifacts/improved_predictions.npz</code>) 
                is not available. Please verify model training artifacts.
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )
        return

    # Filter Controls
    st.markdown(
        textwrap.dedent("""
        <div style="font-size: 13px; font-weight: 700; color: #5E35B1; margin-bottom: 8px;">
            FILTER & SEARCH REAL VALIDATION TRANSACTIONS
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        tier_filter = st.selectbox(
            "Filter by Policy Tier:",
            options=["All Tiers", "ALLOW", "REVIEW", "HIGH RISK"],
            index=0,
        )
    with c2:
        label_filter = st.selectbox(
            "Filter by Ground Truth Label:",
            options=["All Labels", "Fraud (1)", "Legitimate (0)"],
            index=0,
        )
    with c3:
        search_id = st.text_input(
            "Search Transaction ID:",
            placeholder="e.g. 3459434",
        )

    # Apply filters
    filtered_df = df_val.copy()
    if tier_filter != "All Tiers":
        filtered_df = filtered_df[filtered_df["Risk Tier"] == tier_filter]
    if label_filter != "All Labels":
        filtered_df = filtered_df[filtered_df["Actual Label"] == label_filter]
    if search_id.strip():
        try:
            val_id = int(search_id.strip())
            filtered_df = filtered_df[filtered_df["TransactionID"] == val_id]
        except ValueError:
            pass

    st.markdown(
        f"<div class='muted' style='margin-bottom: 12px;'>Showing <strong>{len(filtered_df):,}</strong> matching validation records.</div>",
        unsafe_allow_html=True,
    )

    # Layout: Table on Left, Detail Card on Right
    col_table, col_detail = st.columns([3, 2])

    with col_table:
        st.markdown("### Transaction Records")
        display_cols = ["TransactionID", "Fraud Score", "Risk Tier", "Actual Label"]
        st.dataframe(
            filtered_df[display_cols].head(100),
            use_container_width=True,
            height=420,
        )

    with col_detail:
        st.markdown("### Transaction Inspector")

        # Select transaction ID to inspect
        available_ids = filtered_df["TransactionID"].head(50).tolist()
        if not available_ids:
            st.info("No transactions match the selected filters.")
            return

        selected_id = st.selectbox("Select ID to Inspect:", options=available_ids, index=0)
        txn_row = filtered_df[filtered_df["TransactionID"] == selected_id].iloc[0]

        score = float(txn_row["Fraud Score"])
        tier = str(txn_row["Risk Tier"])
        actual_label = str(txn_row["Actual Label"])

        # Check if detailed SHAP example exists for this transaction
        matched_example = next((ex for ex in examples if ex.get("transaction_id") == selected_id), None)

        card_html = textwrap.dedent(f"""
        <div class="berry-card-white" style="padding: 18px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span class="mono" style="font-size: 16px; font-weight: 700; color: #5E35B1;">#{selected_id}</span>
                {theme.render_decision_chip(tier)}
            </div>
            <div style="margin-bottom: 12px;">
                <div style="font-size: 11.5px; text-transform: uppercase; color: #4B5565; font-weight: 700;">Fraud Risk Score</div>
                <div style="font-size: 28px; font-weight: 800; color: #121926; font-family: 'JetBrains Mono', monospace;">
                    {score * 100:.2f}%
                </div>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 8px 0; border-top: 1px solid #F1F5F9; font-size: 12.5px;">
                <span style="color: #4B5565;">Ground Truth Label:</span>
                <strong style="color: {'#C62828' if 'Fraud' in actual_label else '#00796B'};">{actual_label}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 8px 0; border-top: 1px solid #F1F5F9; font-size: 12.5px;">
                <span style="color: #4B5565;">Policy Tier:</span>
                <strong style="color: #121926;">{tier}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 8px 0; border-top: 1px solid #F1F5F9; font-size: 12.5px;">
                <span style="color: #4B5565;">Cutoff Applied:</span>
                <span class="mono" style="color: #121926;">0.10 (Review) / 0.40 (Decline)</span>
            </div>
        </div>
        """).strip()
        st.markdown(card_html, unsafe_allow_html=True)

        if matched_example:
            with st.expander("Feature Attributions (SHAP)", expanded=True):
                drivers = matched_example.get("top_risk_drivers", [])[:3]
                protect = matched_example.get("top_protective_factors", [])[:3]

                st.markdown("<strong style='color: #C62828; font-size: 12px;'>Top Risk Drivers:</strong>", unsafe_allow_html=True)
                for d in drivers:
                    st.markdown(f"- `{d['feature']}`: value `{d.get('feature_value', 'N/A')}` ({d['shap_attribution']:+.3f})")

                st.markdown("<strong style='color: #00796B; font-size: 12px; margin-top: 6px;'>Protective Factors:</strong>", unsafe_allow_html=True)
                for p in protect:
                    st.markdown(f"- `{p['feature']}`: value `{p.get('feature_value', 'N/A')}` ({p['shap_attribution']:+.3f})")
