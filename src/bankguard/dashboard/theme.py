"""Theme, Design Tokens, and Styling for BankGuard Berry Fintech Dashboard.

Combines:
- The Berry Dashboard architecture (light blue-grey canvas, crisp white cards, vibrant purple & blue gradients).
- Non-negotiable text-visibility rules (WCAG 2.2 AA contrast compliance, ratios >= 4.5:1, primary >= 15:1).
- Sleek navigation pills (radio buttons hidden, smooth hover transitions).
- Rich animated gradients, ambient SVG decorations, and floating icon bubbles.
- Standardized Plotly chart layouts and reusable KPI cards.
"""

from typing import Any, Dict, Optional, Tuple
import plotly.graph_objects as go
import streamlit as st

# Design Tokens (Berry Light Palette + High Contrast FinTech Tokens)
TOKENS = {
    "bg_app": "#EEF2F6",
    "bg_sidebar": "#FFFFFF",
    "bg_panel": "#FFFFFF",
    "bg_panel_solid": "#FFFFFF",
    "bg_panel_hover": "#F8FAFC",
    "bg_input": "#FFFFFF",
    "text_primary": "#121926",
    "text_secondary": "#364152",
    "text_muted": "#4B5565",
    "border_subtle": "#E3E8EF",
    "border_subtle_hex": "#E3E8EF",
    "accent_blue": "#1E88E5",
    "accent_violet": "#5E35B1",
    "risk_low": "#00796B",
    "risk_review": "#C0392B",
    "risk_high": "#C62828",
    "focus_ring": "#5E35B1",
    # Berry Vibrant Gradient Accents
    "berry_purple_start": "#4527A0",
    "berry_purple_mid": "#5E35B1",
    "berry_purple_end": "#7E57C2",
    "berry_blue_start": "#1565C0",
    "berry_blue_mid": "#1E88E5",
    "berry_blue_end": "#29B6F6",
    "berry_purple_tint": "#EDE7F6",
    "berry_blue_tint": "#E3F2FD",
}

# Shorthand references
BG_APP = TOKENS["bg_app"]
BG_SIDEBAR = TOKENS["bg_sidebar"]
BG_PANEL = TOKENS["bg_panel"]
BG_PANEL_SOLID = TOKENS["bg_panel_solid"]
BG_INPUT = TOKENS["bg_input"]
TEXT_PRIMARY = TOKENS["text_primary"]
TEXT_SECONDARY = TOKENS["text_secondary"]
TEXT_MUTED = TOKENS["text_muted"]
BORDER_SUBTLE = TOKENS["border_subtle"]
ACCENT_BLUE = TOKENS["accent_blue"]
ACCENT_VIOLET = TOKENS["accent_violet"]
RISK_LOW = TOKENS["risk_low"]
RISK_REVIEW = TOKENS["risk_review"]
RISK_HIGH = TOKENS["risk_high"]
BERRY_PURPLE_START = TOKENS["berry_purple_start"]
BERRY_BLUE_START = TOKENS["berry_blue_start"]


def parse_hex_color(hex_str: str) -> Tuple[float, float, float]:
    """Parses a 6-digit hex color into normalized RGB floats."""
    clean = hex_str.strip().lstrip("#")
    if len(clean) == 3:
        clean = "".join([c * 2 for c in clean])
    r = int(clean[0:2], 16) / 255.0
    g = int(clean[2:4], 16) / 255.0
    b = int(clean[4:6], 16) / 255.0
    return r, g, b


def relative_luminance(r: float, g: float, b: float) -> float:
    """Calculates relative luminance according to WCAG 2.1/2.2 specifications."""
    def channel_lum(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r_lin = channel_lum(r)
    g_lin = channel_lum(g)
    b_lin = channel_lum(b)
    return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin


def calculate_contrast_ratio(fg_hex: str, bg_hex: str) -> float:
    """Calculates WCAG contrast ratio between foreground and background hex colors."""
    lum1 = relative_luminance(*parse_hex_color(fg_hex))
    lum2 = relative_luminance(*parse_hex_color(bg_hex))
    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)
    return (lighter + 0.05) / (darker + 0.05)


def apply_theme():
    """Injects centralized Berry-inspired stylesheet and layout rules."""
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Global Root & Canvas */
    :root {{
        color-scheme: light;
        --bg-app: {BG_APP};
        --bg-sidebar: {BG_SIDEBAR};
        --bg-panel: {BG_PANEL};
        --bg-input: {BG_INPUT};
        --text-primary: {TEXT_PRIMARY};
        --text-secondary: {TEXT_SECONDARY};
        --text-muted: {TEXT_MUTED};
        --border-subtle: {BORDER_SUBTLE};
        --accent-blue: {ACCENT_BLUE};
        --accent-violet: {ACCENT_VIOLET};
        --risk-low: {RISK_LOW};
        --risk-review: {RISK_REVIEW};
        --risk-high: {RISK_HIGH};
    }}

    .stApp {{
        background-color: {BG_APP} !important;
        color: {TEXT_PRIMARY} !important;
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        line-height: 1.55;
    }}

    /* Main Container Bounds (Generous top padding avoids native header clipping) */
    .block-container {{
        max-width: 1400px !important;
        padding-top: 2rem !important;
        padding-bottom: 3.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }}

    /* Typography */
    h1, h2, h3, h4 {{
        color: {TEXT_PRIMARY} !important;
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }}
    h1 {{ font-size: 26px !important; margin-bottom: 4px !important; }}
    h2 {{ font-size: 20px !important; margin-top: 20px !important; margin-bottom: 12px !important; }}
    h3 {{ font-size: 15px !important; margin-top: 14px !important; margin-bottom: 8px !important; font-weight: 600 !important; }}

    p, span, label, div {{
        color: {TEXT_PRIMARY};
    }}
    .subtext, .caption {{
        color: {TEXT_SECONDARY} !important;
        font-size: 13.5px !important;
    }}
    .muted {{
        color: {TEXT_MUTED} !important;
        font-size: 12.5px !important;
    }}
    .mono {{
        font-family: 'JetBrains Mono', monospace !important;
    }}

    /* Keyframe Animations */
    @keyframes pulseGlow {{
        0% {{ box-shadow: 0 0 0 0 rgba(0, 200, 83, 0.6); }}
        70% {{ box-shadow: 0 0 0 7px rgba(0, 200, 83, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(0, 200, 83, 0); }}
    }}
    @keyframes floatSubtle {{
        0% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-3px); }}
        100% {{ transform: translateY(0px); }}
    }}

    /* Crisp White Panels (Berry Dashboard Style) */
    .glass-card, .berry-card-white {{
        background: #FFFFFF !important;
        border: 1px solid {BORDER_SUBTLE} !important;
        border-radius: 16px !important;
        box-shadow: 0 2px 14px 0 rgba(32, 40, 45, 0.08) !important;
        padding: 22px !important;
        margin-bottom: 16px !important;
        transition: transform 220ms cubic-bezier(0.4, 0, 0.2, 1), box-shadow 220ms ease !important;
    }}
    .glass-card:hover, .berry-card-white:hover {{
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 24px 0 rgba(32, 40, 45, 0.12) !important;
    }}

    /* =========================================================================
       BERRY DASHBOARD HERO CARDS (Vibrant Gradients with Ambient Decorative Circles)
       ========================================================================= */
    .berry-card-purple {{
        background: linear-gradient(135deg, #4527A0 0%, #5E35B1 50%, #7E57C2 100%) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        color: #FFFFFF !important;
        position: relative !important;
        overflow: hidden !important;
        box-shadow: 0 8px 24px 0 rgba(94, 53, 177, 0.3) !important;
        transition: transform 220ms cubic-bezier(0.4, 0, 0.2, 1), box-shadow 220ms ease !important;
        min-height: 190px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
    }}
    .berry-card-purple:hover {{
        transform: translateY(-4px) !important;
        box-shadow: 0 12px 30px 0 rgba(94, 53, 177, 0.42) !important;
    }}
    .berry-card-purple::before {{
        content: "";
        position: absolute;
        top: -30px;
        right: -30px;
        width: 140px;
        height: 140px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.12);
        pointer-events: none;
    }}
    .berry-card-purple::after {{
        content: "";
        position: absolute;
        bottom: -40px;
        right: 30px;
        width: 160px;
        height: 160px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.08);
        pointer-events: none;
    }}

    .berry-card-blue {{
        background: linear-gradient(135deg, #1565C0 0%, #1E88E5 50%, #29B6F6 100%) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        color: #FFFFFF !important;
        position: relative !important;
        overflow: hidden !important;
        box-shadow: 0 8px 24px 0 rgba(30, 136, 229, 0.3) !important;
        transition: transform 220ms cubic-bezier(0.4, 0, 0.2, 1), box-shadow 220ms ease !important;
        min-height: 190px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
    }}
    .berry-card-blue:hover {{
        transform: translateY(-4px) !important;
        box-shadow: 0 12px 30px 0 rgba(30, 136, 229, 0.42) !important;
    }}
    .berry-card-blue::before {{
        content: "";
        position: absolute;
        top: -40px;
        right: -40px;
        width: 170px;
        height: 170px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.12);
        pointer-events: none;
    }}
    .berry-card-blue::after {{
        content: "";
        position: absolute;
        bottom: -20px;
        right: 40px;
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.08);
        pointer-events: none;
    }}

    /* Floating Translucent Icon Bubble in Hero Cards */
    .hero-icon-bubble {{
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.22);
        border: 1px solid rgba(255, 255, 255, 0.35);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        backdrop-filter: blur(8px);
    }}

    /* Compact Side Cards (Right Stack in Berry) */
    .berry-mini-card {{
        background: #FFFFFF !important;
        border: 1px solid {BORDER_SUBTLE} !important;
        border-radius: 14px !important;
        padding: 16px 18px !important;
        display: flex !important;
        align-items: center !important;
        gap: 16px !important;
        box-shadow: 0 2px 10px 0 rgba(32, 40, 45, 0.06) !important;
        transition: transform 200ms ease, box-shadow 200ms ease !important;
        margin-bottom: 12px !important;
    }}
    .berry-mini-card:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px 0 rgba(32, 40, 45, 0.1) !important;
    }}
    .berry-mini-icon {{
        width: 46px;
        height: 46px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        flex-shrink: 0;
    }}

    /* =========================================================================
       SIDEBAR & MODERN PILL NAVIGATION (Hide radio circles completely)
       ========================================================================= */
    section[data-testid="stSidebar"] {{
        background-color: #FFFFFF !important;
        border-right: 1px solid {BORDER_SUBTLE} !important;
        box-shadow: 2px 0 16px 0 rgba(32, 40, 45, 0.04) !important;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {{
        color: {TEXT_PRIMARY} !important;
    }}

    /* Completely eliminate radio buttons and circles */
    div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {{
        display: none !important;
    }}
    div[data-testid="stRadio"] div[role="radiogroup"] {{
        gap: 4px !important;
    }}
    /* Turn each radio label into a clean Berry pill */
    div[data-testid="stRadio"] div[role="radiogroup"] > label {{
        display: flex !important;
        align-items: center !important;
        padding: 10px 14px !important;
        margin-bottom: 4px !important;
        border-radius: 10px !important;
        background-color: transparent !important;
        color: {TEXT_SECONDARY} !important;
        font-weight: 500 !important;
        font-size: 13.5px !important;
        cursor: pointer !important;
        transition: all 180ms cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: 1px solid transparent !important;
    }}
    div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {{
        background-color: #EDE7F6 !important;
        color: #5E35B1 !important;
        transform: translateX(4px) !important;
    }}
    /* Active / Checked Berry pill */
    div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"],
    div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {{
        background-color: #EDE7F6 !important;
        color: #5E35B1 !important;
        font-weight: 700 !important;
        border: 1px solid rgba(94, 53, 177, 0.25) !important;
        box-shadow: 0 2px 8px rgba(94, 53, 177, 0.12) !important;
    }}
    div[data-testid="stRadio"] div[role="radiogroup"] > label p {{
        color: inherit !important;
        font-weight: inherit !important;
        font-size: 13.5px !important;
        margin: 0 !important;
    }}

    /* Inputs, Selectboxes & Form Controls */
    div[data-baseweb="input"],
    div[data-baseweb="select"] > div,
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[role="combobox"] {{
        background-color: #FFFFFF !important;
        color: {TEXT_PRIMARY} !important;
        border: 1px solid #CFD8DC !important;
        border-radius: 10px !important;
        font-size: 13.5px !important;
    }}
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="select"] > div:focus-within {{
        border-color: #5E35B1 !important;
        box-shadow: 0 0 0 2px rgba(94, 53, 177, 0.2) !important;
    }}

    /* Primary Buttons */
    .stButton > button {{
        background: #5E35B1 !important;
        color: #FFFFFF !important;
        border: 1px solid #5E35B1 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        padding: 8px 18px !important;
        transition: all 180ms ease !important;
        box-shadow: 0 2px 8px rgba(94, 53, 177, 0.2) !important;
    }}
    .stButton > button:hover {{
        background: #4527A0 !important;
        border-color: #4527A0 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px rgba(94, 53, 177, 0.35) !important;
        transform: translateY(-1px) !important;
    }}

    /* Tables & Feeds */
    table {{
        background-color: #FFFFFF !important;
        border-collapse: collapse !important;
        width: 100% !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid {BORDER_SUBTLE} !important;
        font-size: 13px !important;
    }}
    th {{
        background-color: #F8FAFC !important;
        color: {TEXT_PRIMARY} !important;
        font-weight: 600 !important;
        padding: 12px 16px !important;
        border-bottom: 1px solid {BORDER_SUBTLE} !important;
        text-align: left !important;
    }}
    td {{
        padding: 10px 16px !important;
        border-bottom: 1px solid #F1F5F9 !important;
        color: {TEXT_SECONDARY} !important;
    }}
    tr:nth-child(even) {{
        background-color: #FAFCFE !important;
    }}
    tr:hover td {{
        background-color: #F1F5F9 !important;
        color: {TEXT_PRIMARY} !important;
    }}

    /* Live Watchlist Feed Row (Berry Style) */
    .feed-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 14px;
        border-bottom: 1px solid #F1F5F9;
        transition: background 150ms ease;
        border-radius: 8px;
    }}
    .feed-row:hover {{
        background: #F8FAFC;
    }}

    /* Top Persistent Status Bar */
    .top-status-bar {{
        background: #FFFFFF;
        border: 1px solid {BORDER_SUBTLE};
        border-radius: 12px;
        padding: 10px 18px;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 10px 0 rgba(32, 40, 45, 0.05);
    }}
    .status-chip {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        background: #F4F6FB;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        font-size: 11.5px;
        font-family: 'JetBrains Mono', monospace;
        color: {TEXT_PRIMARY};
    }}

    /* Decision Chips (Tiers) */
    .chip {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 14px;
        font-size: 11px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }}
    .chip-allow {{
        background: #E0F2F1;
        color: {RISK_LOW};
        border: 1px solid #A7F3D0;
    }}
    .chip-review {{
        background: #FBE9E7;
        color: {RISK_REVIEW};
        border: 1px solid #FFCCBC;
    }}
    .chip-block {{
        background: #FFEBEE;
        color: {RISK_HIGH};
        border: 1px solid #FFCDD2;
    }}

    /* Live Pulse Dot */
    .pulse-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #00C853;
        animation: pulseGlow 1.8s infinite;
        display: inline-block;
    }}

    /* Notice callouts */
    .callout-amber {{
        background: #FFF8E1;
        border: 1px solid #FFE082;
        border-left: 4px solid #FFA000;
        border-radius: 10px;
        padding: 12px 16px;
        color: #4A3A00;
        font-size: 13px;
        margin-bottom: 16px;
    }}
    .callout-blue {{
        background: #E3F2FD;
        border: 1px solid #BBDEFB;
        border-left: 4px solid #1976D2;
        border-radius: 10px;
        padding: 12px 16px;
        color: #0D47A1;
        font-size: 13px;
        margin-bottom: 16px;
    }}

    /* Footer */
    .dashboard-footer {{
        border-top: 1px solid {BORDER_SUBTLE};
        padding-top: 24px;
        margin-top: 40px;
        font-size: 12px;
        color: {TEXT_MUTED};
        text-align: center;
        line-height: 1.6;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_sidebar_header():
    """Renders Berry-style brand header in the sidebar."""
    html = f"""
    <div style="padding: 10px 4px 18px 4px; border-bottom: 1px solid {BORDER_SUBTLE}; margin-bottom: 16px;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="width: 42px; height: 42px; border-radius: 12px; background: linear-gradient(135deg, #5E35B1 0%, #1E88E5 100%); display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(94, 53, 177, 0.35);">
                <span style="font-size: 20px;">🛡️</span>
            </div>
            <div>
                <div style="font-size: 20px; font-weight: 800; color: #121926; letter-spacing: -0.02em;">
                    BankGuard
                </div>
                <div style="font-size: 11px; font-weight: 700; color: #5E35B1; letter-spacing: 0.05em;">
                    RISK INTELLIGENCE
                </div>
            </div>
        </div>
        <div style="font-size: 11.5px; color: {TEXT_MUTED}; margin-top: 8px;">
            Berry FinTech &bull; Production Engine
        </div>
    </div>
    """
    st.sidebar.markdown(html, unsafe_allow_html=True)


def render_sidebar_footer():
    """Renders Berry-style system card at bottom of sidebar."""
    html = f"""
    <div style="margin-top: 24px; padding: 14px; background: linear-gradient(135deg, #EDE7F6 0%, #E3F2FD 100%); border: 1px solid #D1C4E9; border-radius: 14px;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
            <span class="pulse-dot"></span>
            <span style="font-size: 11.5px; font-weight: 700; color: #5E35B1; letter-spacing: 0.02em;">ENGINE ACTIVE</span>
        </div>
        <div style="font-size: 12px; color: {TEXT_SECONDARY}; margin-bottom: 8px;">
            Model: <strong>LightGBM 4.7.0</strong><br/>
            Policy: <strong style="color: #5E35B1;">Three-Tier (&tau;=0.22)</strong>
        </div>
        <div style="background: rgba(255,255,255,0.6); border-radius: 6px; height: 6px; overflow: hidden;">
            <div style="background: linear-gradient(90deg, #5E35B1, #1E88E5); width: 88%; height: 100%;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 10px; color: {TEXT_MUTED}; margin-top: 4px;">
            <span>Queue Capacity</span>
            <span>2.96% / 3.0%</span>
        </div>
    </div>
    """
    st.sidebar.markdown(html, unsafe_allow_html=True)


def render_top_bar():
    """Renders persistent top status strip across all dashboard pages."""
    html = f"""
    <div class="top-status-bar">
        <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
            <span class="status-chip"><span class="pulse-dot"></span> Benchmark: <strong>IEEE-CIS</strong></span>
            <span class="status-chip">Model: <strong>LightGBM Classifier</strong></span>
            <span class="status-chip">Validation: <strong>118,108 Txns (OOT)</strong></span>
        </div>
        <div>
            <span style="font-size: 11.5px; color: {TEXT_MUTED};">
                Demonstration Portfolio System
            </span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_kpi_card(title: str, value: str, caption: str, meaning: str, accent_color: Optional[str] = None):
    """Renders a standard Berry KPI card with high contrast and deliberate hierarchy."""
    border_accent = f"border-left: 4px solid {accent_color} !important;" if accent_color else ""
    html = f"""
    <div class="berry-card-white" style="padding: 16px 18px; {border_accent}">
        <div style="font-size: 11.5px; font-weight: 700; color: #4B5565; text-transform: uppercase; letter-spacing: 0.04em;">
            {title}
        </div>
        <div style="font-size: 26px; font-weight: 800; color: #121926; margin-top: 4px; font-family: 'JetBrains Mono', monospace; line-height: 1.15;">
            {value}
        </div>
        <div style="font-size: 12.5px; color: #364152; font-weight: 500; margin-top: 4px;">
            {caption}
        </div>
        <div style="font-size: 11px; color: #4B5565; margin-top: 6px; padding-top: 6px; border-top: 1px solid #F1F5F9;">
            {meaning}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_decision_chip(tier: str) -> str:
    """Returns HTML for decision tier chip with icon shape, label, and high-contrast color."""
    t_clean = str(tier).lower().strip()
    if t_clean in ("allow", "approved", "low_risk"):
        return '<span class="chip chip-allow">● ALLOW</span>'
    elif t_clean in ("review", "suspicious", "amber"):
        return '<span class="chip chip-review">▲ REVIEW</span>'
    else:
        return '<span class="chip chip-block">⬢ HIGH RISK</span>'


def render_footer():
    """Renders mandatory calm footer on every page."""
    html = f"""
    <div class="dashboard-footer">
        <strong>BankGuard Fraud Risk Intelligence</strong> &bull; Evaluated on the public IEEE-CIS Fraud Detection Benchmark.<br/>
        Demonstration application for explainable risk assessment and decisioning architectures. Does not authorize live payment rails.
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def get_plotly_layout(height: int = 380, title: Optional[str] = None) -> Dict[str, Any]:
    """Returns standardized Plotly layout adhering to Berry clean light theme."""
    return {
        "height": height,
        "title": {
            "text": title or "",
            "font": {"family": "Plus Jakarta Sans, Inter, sans-serif", "size": 14, "color": TEXT_PRIMARY},
            "x": 0.0,
            "y": 0.98,
        },
        "paper_bgcolor": "#FFFFFF",
        "plot_bgcolor": "#FFFFFF",
        "margin": {"l": 50, "r": 30, "t": 48 if title else 25, "b": 45},
        "font": {"family": "Plus Jakarta Sans, Inter, sans-serif", "size": 12, "color": TEXT_SECONDARY},
        "xaxis": {
            "gridcolor": "#F1F5F9",
            "gridwidth": 1,
            "zeroline": False,
            "tickfont": {"size": 11, "color": TEXT_MUTED},
            "titlefont": {"size": 12, "color": TEXT_PRIMARY},
        },
        "yaxis": {
            "gridcolor": "#F1F5F9",
            "gridwidth": 1,
            "zeroline": False,
            "tickfont": {"size": 11, "color": TEXT_MUTED},
            "titlefont": {"size": 12, "color": TEXT_PRIMARY},
        },
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
            "font": {"size": 11, "color": TEXT_SECONDARY},
        },
    }
