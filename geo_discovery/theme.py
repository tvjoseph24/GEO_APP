"""WPP Production theme — single source of truth for colour, type and widget styling.

Every colour in the app and in the exported charts resolves here. Keep the tokens below
in step with .streamlit/config.toml (PRIMARY / BG / SURFACE / INK).

NOTE: the hex values are sampled from the Creative Studio UI. Replace with the official
WPP Production brand values when available — this is the only file that needs editing.
"""
import streamlit as st

# ---------------------------------------------------------------- tokens
PRIMARY       = "#1F40D8"   # royal blue — links, active state, key figures
PRIMARY_DARK  = "#152C99"
PRIMARY_TINT  = "#EAF0FE"   # active pill fill, card fill
INK           = "#1A1A2E"   # body text
INK_MUTED     = "#6B7280"   # captions, secondary text
BG            = "#FFFFFF"
SURFACE       = "#F4F5F7"   # cards, sidebar
BORDER        = "#E2E4E9"
BORDER_STRONG = "#C9CDD6"

GOOD    = "#00A36A"
WARN    = "#E8A33D"
DANGER  = "#D64545"
NEUTRAL = "#8C9196"

# logo gradient — reserve for the title rule and accents only
GRAD = ["#E6007E", "#7B2FF7", "#1F40D8", "#00C389"]

FONT = ("'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', "
        "Roboto, Helvetica, Arial, sans-serif")

# chart palette — imported by report/charts.py so app and exports match
CHART = {"brand": PRIMARY, "alt": NEUTRAL, "good": GOOD, "warn": WARN,
         "bad": DANGER, "grey": BORDER, "ink": INK, "muted": INK_MUTED}

# ---------------------------------------------------------------- labels
# No emoji. Numbered so pipeline order is legible at a glance.
TABS = ["1 · Discover", "2 · Audit", "3 · Benchmark",
        "4 · Experiment", "5 · Optimise", "6 · Report"]

# pipeline status markers — geometric, not emoji
STATUS = {"done": "●", "ready": "○", "locked": "—"}

# ---------------------------------------------------------------- css
_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, .stCaption,
button, input, textarea, select {{
    font-family: {FONT};
}}
body, .stMarkdown {{ color: {INK}; }}

/* ---------------- headings ---------------- */
h1 {{ font-weight: 600 !important; letter-spacing: -0.025em;
      color: {INK} !important; font-size: 2.1rem !important; }}
h2 {{ font-weight: 600 !important; letter-spacing: -0.012em;
      color: {INK} !important; font-size: 1.45rem !important;
      margin-top: 1.6rem !important; }}
h3 {{ font-weight: 500 !important; color: {INK} !important;
      font-size: 1.13rem !important; }}

/* gradient accent rule under the app title */
hr.geo-rule {{
    height: 3px; width: 68px; border: 0; border-radius: 2px;
    margin: .35rem 0 1.1rem 0;
    background: linear-gradient(90deg, {GRAD[0]}, {GRAD[1]}, {GRAD[2]}, {GRAD[3]});
}}
p.geo-sub {{ color: {INK_MUTED}; font-size: .95rem; margin: -.5rem 0 1.4rem 0; }}

/* ---------------- TABS as pills ---------------- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px; background: transparent; border-bottom: none;
    padding: 2px 0 22px 0; flex-wrap: wrap;
}}
.stTabs [data-baseweb="tab"] {{
    height: auto; padding: 9px 26px; background: {BG};
    border: 1px solid {BORDER}; border-radius: 999px;
    color: {INK_MUTED}; font-weight: 500;
    transition: background .15s ease, border-color .15s ease, color .15s ease;
}}
.stTabs [data-baseweb="tab"]:hover {{
    background: {SURFACE}; color: {INK}; border-color: {BORDER_STRONG};
}}
.stTabs [aria-selected="true"] {{
    background: {PRIMARY_TINT} !important; border-color: {PRIMARY} !important;
    color: {PRIMARY} !important; font-weight: 600 !important;
}}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {{ display: none !important; }}
.stTabs [data-baseweb="tab-list"] button p {{
    font-size: .93rem !important; margin: 0 !important;
}}

/* ---------------- buttons as pills ---------------- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
    border-radius: 999px; border: 1px solid {BORDER};
    padding: .45rem 1.3rem; font-weight: 500; font-size: .9rem;
    background: {BG}; color: {INK};
    transition: background .15s ease, border-color .15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover,
.stFormSubmitButton > button:hover {{
    background: {SURFACE}; border-color: {BORDER_STRONG}; color: {INK};
}}
.stButton > button[kind="primary"], .stFormSubmitButton > button {{
    background: {PRIMARY}; border-color: {PRIMARY}; color: #FFFFFF;
}}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button:hover {{
    background: {PRIMARY_DARK}; border-color: {PRIMARY_DARK}; color: #FFFFFF;
}}
.stButton > button:disabled {{ opacity: .45; }}

/* ---------------- metrics as cards ---------------- */
[data-testid="stMetric"] {{
    background: {SURFACE}; border: 1px solid {BORDER};
    border-radius: 14px; padding: 14px 18px;
}}
[data-testid="stMetricValue"] {{
    font-weight: 600 !important; font-size: 1.65rem !important;
    color: {PRIMARY} !important;
}}
[data-testid="stMetricLabel"] p {{
    font-size: .78rem !important; font-weight: 500 !important;
    color: {INK_MUTED} !important; text-transform: uppercase;
    letter-spacing: .04em;
}}
[data-testid="stMetricDelta"] {{ font-size: .8rem !important; }}

/* ---------------- sidebar ---------------- */
[data-testid="stSidebar"] {{
    background: {SURFACE}; border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
    font-size: .8rem !important; font-weight: 600 !important;
    text-transform: uppercase; letter-spacing: .08em;
    color: {INK_MUTED} !important; margin-top: 1.4rem !important;
}}
[data-testid="stSidebar"] [data-testid="stMetric"] {{
    background: {BG}; padding: 10px 14px;
}}

/* ---------------- expanders ---------------- */
[data-testid="stExpander"], details[data-testid="stExpander"] {{
    border: 1px solid {BORDER} !important; border-radius: 12px !important;
    background: {BG}; margin-bottom: .55rem;
}}
[data-testid="stExpander"] summary {{
    font-weight: 500; font-size: .93rem; padding: .6rem .95rem;
}}
[data-testid="stExpander"] summary:hover {{ color: {PRIMARY}; }}
[data-testid="stExpander"] summary p {{ font-size: .93rem !important; }}

/* ---------------- callouts ---------------- */
[data-testid="stAlert"], .stAlert {{
    border-radius: 12px; border-left-width: 4px !important; font-size: .9rem;
}}

/* ---------------- tables ---------------- */
[data-testid="stDataFrame"], [data-testid="stTable"] {{
    border: 1px solid {BORDER}; border-radius: 12px; overflow: hidden;
}}

/* ---------------- inputs ---------------- */
.stTextInput input, .stTextArea textarea, .stNumberInput input,
[data-baseweb="select"] > div, [data-baseweb="input"] {{
    border-radius: 10px !important; border-color: {BORDER} !important;
}}
.stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {{
    border-color: {PRIMARY} !important;
    box-shadow: 0 0 0 2px {PRIMARY_TINT} !important;
}}
.stTextInput label, .stTextArea label, .stNumberInput label,
.stSelectbox label, .stMultiSelect label, .stSlider label, .stRadio label {{
    font-size: .85rem !important; font-weight: 500 !important; color: {INK} !important;
}}

/* ---------------- progress, slider, toggles ---------------- */
[data-testid="stProgress"] > div > div > div > div {{ background: {PRIMARY}; }}
[data-baseweb="slider"] [role="slider"] {{ background: {PRIMARY} !important; }}
[data-testid="stSidebar"] [data-baseweb="checkbox"] svg,
[data-baseweb="checkbox"] svg {{ fill: {PRIMARY}; }}

/* multiselect / tag chips */
[data-baseweb="tag"] {{
    background: {PRIMARY_TINT} !important; color: {PRIMARY} !important;
    border-radius: 999px !important; border: 1px solid {PRIMARY} !important;
}}

/* ---------------- captions ---------------- */
.stCaption, [data-testid="stCaptionContainer"] p {{
    color: {INK_MUTED} !important; font-size: .82rem !important;
}}

/* ---------------- chrome removal ---------------- */
#MainMenu, footer, [data-testid="stDecoration"],
[data-testid="stToolbar"] {{ visibility: hidden; height: 0; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }}
</style>
"""

# ---------------------------------------------------------------- api
def inject():
    """Call once, immediately after st.set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)

def title(text, subtitle=""):
    """App title with the gradient accent rule."""
    st.markdown(f"<h1>{text}</h1>", unsafe_allow_html=True)
    st.markdown("<hr class='geo-rule'>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p class='geo-sub'>{subtitle}</p>", unsafe_allow_html=True)

def section(text, subtitle=""):
    """Phase header for a tab — replaces st.header(emoji + 'Module N: X')."""
    st.markdown(f"<h2 style='margin-top:.2rem'>{text}</h2>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p class='geo-sub' style='margin-top:-.7rem'>{subtitle}</p>",
                    unsafe_allow_html=True)