"""BankGuard Berry FinTech Dashboard — Main Application Entry Point.

Implements:
- Berry-inspired fintech dashboard architecture with vibrant hero cards and ambient SVGs.
- Clean white cards with soft elevation shadows and smooth transitions.
- Non-negotiable text visibility and WCAG 2.2 AA contrast compliance.
- 7 modular intelligence pages backed by real validation data.
"""

import textwrap
import streamlit as st
from src.bankguard.dashboard import theme
from src.bankguard.dashboard.pages import (
    page_data_quality,
    page_explainability,
    page_explorer,
    page_model_performance,
    page_overview,
    page_risk_analysis,
    page_system_health,
)

# Page configuration
st.set_page_config(
    page_title="BankGuard — Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    # Inject Centralized Berry Theme & Stylesheet
    theme.apply_theme()

    # Sidebar Header
    theme.render_sidebar_header()

    # Grouped Navigation Label (Berry Dashboard Style)
    st.sidebar.markdown(
        textwrap.dedent("""
        <div style="font-size: 11px; font-weight: 700; color: #5E35B1; letter-spacing: 0.08em; margin-bottom: 6px; padding-left: 6px; text-transform: uppercase;">
            DASHBOARD &amp; INTELLIGENCE
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )

    pages = {
        "📊  Overview": page_overview.render,
        "🔍  Transaction Explorer": page_explorer.render,
        "⚖️  Risk Analysis": page_risk_analysis.render,
        "📈  Model Performance": page_model_performance.render,
        "💡  Explainable AI": page_explainability.render,
        "🛡️  Data Quality": page_data_quality.render,
        "🩺  System Health": page_system_health.render,
    }

    selected_page_label = st.sidebar.radio(
        "Navigation",
        options=list(pages.keys()),
        index=0,
        label_visibility="collapsed",
    )

    # Sidebar Berry-Style System Card
    theme.render_sidebar_footer()

    # Top Status Bar
    theme.render_top_bar()

    # Render Selected Page
    renderer = pages[selected_page_label]
    renderer()

    # Mandatory Footer
    theme.render_footer()


if __name__ == "__main__":
    main()
