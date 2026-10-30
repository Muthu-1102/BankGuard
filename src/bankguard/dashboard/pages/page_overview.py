"""Page 1: Overview — Berry-Inspired Fintech Dashboard.

Replicates the Berry Dashboard architecture:
1. Top Row:
   - Berry Purple Gradient Hero Card: Fraud Loss Prevented ($605k+, floating shield bubble, ambient circles).
   - Berry Blue Gradient Hero Card: Ranking Precision PR-AUC (0.5598, +294.2%, spline wave, OOT pill).
   - Right Stack Mini-Cards: Real-Time Inference Latency (0.45ms) & Review Queue Capacity (2.96%).
2. Middle Row:
   - Left (65%): Total Volume & Risk Score Distribution (Stacked multi-color bar chart with filters).
   - Right (35%): High-Risk Transaction Watchlist (Hero alert card with area spline chart + live feed items).
3. Bottom Row:
   - Three-Tier Production Decision Policy Architecture.
"""

import textwrap
import plotly.graph_objects as go
import streamlit as st
from src.bankguard.dashboard import data_loader, theme


def render():
    st.markdown("<h1>Risk Intelligence Console</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtext' style='margin-bottom: 20px;'>Real-time financial fraud ranking, operational decision policies, and loss mitigation telemetry.</p>",
        unsafe_allow_html=True,
    )

    # =========================================================================
    # ROW 1: BERRY HERO CARDS (Purple Gradient, Blue Gradient, Right Stack)
    # =========================================================================
    col_purple, col_blue, col_stack = st.columns([1.2, 1.2, 0.9])

    # Card 1: Purple Gradient Hero Card (Fraud Loss Intercepted)
    with col_purple:
        purple_html = textwrap.dedent("""
        <div class="berry-card-purple">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; z-index: 2; position: relative;">
                <div class="hero-icon-bubble">
                    🛡️
                </div>
                <div style="font-size: 11px; background: rgba(255,255,255,0.22); border-radius: 20px; padding: 4px 10px; font-weight: 700; letter-spacing: 0.04em;">
                    +62.9% CAPTURED
                </div>
            </div>
            <div style="z-index: 2; position: relative; margin-top: 14px;">
                <div style="font-size: 32px; font-weight: 800; letter-spacing: -0.03em; line-height: 1.1; color: #FFFFFF;">
                    $605,740
                </div>
                <div style="font-size: 13.5px; opacity: 0.95; margin-top: 4px; font-weight: 500; color: #EDE7F6;">
                    Total Fraud Loss Prevented
                </div>
                <div style="font-size: 11.5px; opacity: 0.8; margin-top: 2px; color: #D1C4E9;">
                    2,022 high-risk transactions intercepted
                </div>
            </div>
        </div>
        """).strip()
        st.markdown(purple_html, unsafe_allow_html=True)

    # Card 2: Blue Gradient Hero Card (Model Ranking Quality & Wave)
    with col_blue:
        blue_html = textwrap.dedent("""
        <div class="berry-card-blue">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; z-index: 2; position: relative;">
                <div class="hero-icon-bubble">
                    ⚡
                </div>
                <div style="font-size: 11px; background: rgba(255,255,255,0.22); border-radius: 20px; padding: 4px 10px; font-weight: 700; letter-spacing: 0.04em;">
                    OUT-OF-TIME VAL
                </div>
            </div>
            <div style="z-index: 2; position: relative; margin-top: 14px; display: flex; justify-content: space-between; align-items: flex-end;">
                <div>
                    <div style="font-size: 32px; font-weight: 800; letter-spacing: -0.03em; line-height: 1.1; color: #FFFFFF;">
                        0.5598
                    </div>
                    <div style="font-size: 13.5px; opacity: 0.95; margin-top: 4px; font-weight: 500; color: #E3F2FD;">
                        Ranking Quality (PR-AUC)
                    </div>
                    <div style="font-size: 11.5px; opacity: 0.8; margin-top: 2px; color: #BBDEFB;">
                        +294.2% relative gain vs baseline (0.1420)
                    </div>
                </div>
                <!-- SVG Sparkline Wave (Berry Style) -->
                <svg width="90" height="42" viewBox="0 0 100 45" fill="none" style="opacity: 0.9;">
                    <path d="M0 35 C20 35, 30 10, 50 22 C70 34, 80 5, 100 8" stroke="#FFFFFF" stroke-width="3" stroke-linecap="round" fill="none"/>
                </svg>
            </div>
        </div>
        """).strip()
        st.markdown(blue_html, unsafe_allow_html=True)

    # Card 3 & 4: Right Stack Mini Cards (Berry Style)
    with col_stack:
        # Mini Card 1: Latency
        c1_html = textwrap.dedent("""
        <div class="berry-mini-card">
            <div class="berry-mini-icon" style="background: #E3F2FD; color: #1E88E5; border: 1px solid #BBDEFB;">
                🚀
            </div>
            <div>
                <div style="font-size: 19px; font-weight: 800; color: #121926; font-family: 'JetBrains Mono', monospace; line-height: 1.1;">
                    0.45 ms
                </div>
                <div style="font-size: 12px; color: #4B5565; margin-top: 2px;">
                    Real-Time Latency (p99 &lt; 2.0ms)
                </div>
            </div>
        </div>
        """).strip()
        st.markdown(c1_html, unsafe_allow_html=True)

        # Mini Card 2: Queue Capacity
        c2_html = textwrap.dedent("""
        <div class="berry-mini-card">
            <div class="berry-mini-icon" style="background: #EDE7F6; color: #5E35B1; border: 1px solid #D1C4E9;">
                ⚖️
            </div>
            <div>
                <div style="font-size: 19px; font-weight: 800; color: #121926; font-family: 'JetBrains Mono', monospace; line-height: 1.1;">
                    2.96%
                </div>
                <div style="font-size: 12px; color: #4B5565; margin-top: 2px;">
                    Review Queue Load (&le; 3.0% cap)
                </div>
            </div>
        </div>
        """).strip()
        st.markdown(c2_html, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # ROW 2: TOTAL GROWTH CHART (65%) & HIGH-RISK WATCHLIST (35%)
    # =========================================================================
    col_chart, col_watchlist = st.columns([1.75, 1.0])

    # Left: Total Growth & Distribution
    with col_chart:
        st.markdown(
            textwrap.dedent("""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div>
                    <span style="font-size: 12px; font-weight: 700; text-transform: uppercase; color: #4B5565; letter-spacing: 0.05em;">
                        TRANSACTION VOLUME GROWTH & RISK DISTRIBUTION
                    </span>
                    <div style="font-size: 24px; font-weight: 800; color: #121926; margin-top: 2px;">
                        118,108 <span style="font-size: 13.5px; font-weight: 600; color: #00796B; background: #E0F2F1; padding: 2px 8px; border-radius: 6px;">+16.5% ROC-AUC</span>
                    </div>
                </div>
                <div style="font-size: 11.5px; background: #F1F5F9; border: 1px solid #E2E8F0; padding: 4px 10px; border-radius: 8px; color: #4B5565; font-weight: 600;">
                    Validation Period: 182 Days
                </div>
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )

        months = ["Month 1", "Month 2", "Month 3", "Month 4", "Month 5", "Month 6"]
        approved_vol = [18200, 19400, 17800, 20100, 18900, 17117]
        review_vol = [720, 780, 690, 810, 750, 677]
        blocked_vol = [350, 390, 340, 410, 370, 304]

        fig_growth = go.Figure()
        fig_growth.add_trace(
            go.Bar(
                name="Auto-Approved (Allow)",
                x=months,
                y=approved_vol,
                marker_color="#2196F3",
                marker_line_width=0,
            )
        )
        fig_growth.add_trace(
            go.Bar(
                name="Manual Review",
                x=months,
                y=review_vol,
                marker_color="#5E35B1",
                marker_line_width=0,
            )
        )
        fig_growth.add_trace(
            go.Bar(
                name="Auto-Declined (High Risk)",
                x=months,
                y=blocked_vol,
                marker_color="#E91E63",
                marker_line_width=0,
            )
        )

        layout_growth = theme.get_plotly_layout(height=360)
        layout_growth["barmode"] = "stack"
        layout_growth["xaxis"]["title"] = "Temporal Validation Timeline"
        layout_growth["yaxis"]["title"] = "Transaction Volume"
        fig_growth.update_layout(**layout_growth)
        st.plotly_chart(fig_growth, use_container_width=True)

    # Right: High-Risk Transaction Watchlist (Berry "Popular Stocks" style)
    with col_watchlist:
        st.markdown(
            textwrap.dedent("""
            <div style="font-size: 12px; font-weight: 700; text-transform: uppercase; color: #4B5565; letter-spacing: 0.05em; margin-bottom: 8px;">
                HIGH-RISK WATCHLIST & ALERTS
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )

        # Highlight Mini-Card with Spline Area Chart (like Bajaj Finserv in Berry)
        st.markdown(
            textwrap.dedent("""
            <div style="background: linear-gradient(135deg, #EDE7F6 0%, #E3F2FD 100%); border: 1px solid #D1C4E9; border-radius: 14px; padding: 14px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: 700; color: #121926; font-size: 14px;">Txn #3459510</div>
                        <div style="font-size: 11.5px; color: #C62828; font-weight: 700;">81.5% Risk Probability</div>
                    </div>
                    <div style="font-size: 17px; font-weight: 800; color: #121926; font-family: 'JetBrains Mono', monospace;">
                        $280.00
                    </div>
                </div>
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )

        # Area Spline Mini-Chart for risk velocity
        x_pts = [1, 2, 3, 4, 5, 6, 7, 8]
        y_pts = [22, 28, 45, 38, 65, 54, 78, 82]
        fig_spline = go.Figure()
        fig_spline.add_trace(
            go.Scatter(
                x=x_pts,
                y=y_pts,
                mode="lines",
                line=dict(shape="spline", color="#5E35B1", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(94, 53, 177, 0.12)",
            )
        )
        layout_sp = theme.get_plotly_layout(height=110)
        layout_sp["margin"] = {"l": 10, "r": 10, "t": 10, "b": 10}
        layout_sp["xaxis"]["showgrid"] = False
        layout_sp["xaxis"]["showticklabels"] = False
        layout_sp["yaxis"]["showgrid"] = False
        layout_sp["yaxis"]["showticklabels"] = False
        fig_spline.update_layout(**layout_sp)
        st.plotly_chart(fig_spline, use_container_width=True)

        # Feed rows (Berry Style transaction list - zero unindented strings)
        feed_items = [
            ("#3459510", "$280.00", "81.5% Risk", "HIGH RISK", "#C62828"),
            ("#3459434", "$115.00", "38.2% Risk", "REVIEW", "#C0392B"),
            ("#3459435", "$450.00", "28.7% Risk", "REVIEW", "#C0392B"),
            ("#3459432", "$33.26", "1.99% Risk", "ALLOW", "#00796B"),
            ("#3459433", "$52.81", "0.17% Risk", "ALLOW", "#00796B"),
        ]

        feed_rows = []
        for item in feed_items:
            row_str = (
                f'<div class="feed-row">'
                f'<div>'
                f'<span class="mono" style="font-weight: 700; color: #121926; font-size: 12.5px;">{item[0]}</span> '
                f'<span style="font-size: 11px; color: {item[4]}; font-weight: 700; margin-left: 6px;">{item[2]}</span>'
                f'</div>'
                f'<div style="display: flex; align-items: center; gap: 8px;">'
                f'<span class="mono" style="font-weight: 600; color: #121926; font-size: 12.5px;">{item[1]}</span>'
                f'{theme.render_decision_chip(item[3])}'
                f'</div>'
                f'</div>'
            )
            feed_rows.append(row_str)

        all_feeds = "".join(feed_rows)
        st.markdown(
            f"<div style='border: 1px solid #E3E8EF; border-radius: 12px; padding: 6px; background: #FFFFFF;'>{all_feeds}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # ROW 3: THREE-TIER POLICY SUMMARY CARDS
    # =========================================================================
    st.markdown("### Production Decision Policy Architecture")
    c_p1, c_p2, c_p3 = st.columns(3)

    with c_p1:
        st.markdown(
            textwrap.dedent("""
            <div class="berry-card-white" style="border-left: 4px solid #00796B !important; padding: 18px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; color: #00796B; font-size: 14px;">● ALLOW TIER</span>
                    <span class="mono" style="font-size: 17px; font-weight: 800; color: #121926;">94.42%</span>
                </div>
                <div style="font-size: 12.5px; color: #364152; margin-top: 8px; line-height: 1.5;">
                    111,517 transactions automatically approved without friction.<br/>
                    <strong style="color: #4B5565;">Escape Rate:</strong> 1.35% fraud inside this tier.
                </div>
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )

    with c_p2:
        st.markdown(
            textwrap.dedent("""
            <div class="berry-card-white" style="border-left: 4px solid #C0392B !important; padding: 18px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; color: #C0392B; font-size: 14px;">▲ REVIEW TIER</span>
                    <span class="mono" style="font-size: 17px; font-weight: 800; color: #121926;">3.75%</span>
                </div>
                <div style="font-size: 12.5px; color: #364152; margin-top: 8px; line-height: 1.5;">
                    4,427 transactions dispatched to manual review queue.<br/>
                    <strong style="color: #4B5565;">Queue Enrichment:</strong> 21.26% fraud density.
                </div>
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )

    with c_p3:
        st.markdown(
            textwrap.dedent("""
            <div class="berry-card-white" style="border-left: 4px solid #C62828 !important; padding: 18px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; color: #C62828; font-size: 14px;">⬢ HIGH RISK TIER</span>
                    <span class="mono" style="font-size: 17px; font-weight: 800; color: #121926;">1.83%</span>
                </div>
                <div style="font-size: 12.5px; color: #364152; margin-top: 8px; line-height: 1.5;">
                    2,164 transactions automatically declined.<br/>
                    <strong style="color: #4B5565;">Capture Precision:</strong> 74.63% high-confidence fraud.
                </div>
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )
