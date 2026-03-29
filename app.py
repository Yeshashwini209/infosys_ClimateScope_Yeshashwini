import hashlib
import json
import os
import secrets
import streamlit as st
from io import BytesIO
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity

from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import kruskal

st.set_page_config(page_title="ClimateScope Global Weather Intelligence System", layout="wide")

_USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "registered_users.json")


def _get_demo_credentials():
    """Optional: set [auth] username / password in .streamlit/secrets.toml."""
    try:
        return str(st.secrets["auth"]["username"]), str(st.secrets["auth"]["password"])
    except Exception:
        return "admin", "climate2026"


def _load_users() -> dict:
    if not os.path.isfile(_USERS_FILE):
        return {}
    try:
        with open(_USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_users(users: dict) -> None:
    with open(_USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def _verify_demo_login(username: str, password: str) -> bool:
    demo_user, demo_pw = _get_demo_credentials()
    return username.strip() == demo_user and password == demo_pw


def _get_user_key_for_login(login: str) -> str | None:
    login_clean = login.strip().lower()
    if not login_clean:
        return None
    users = _load_users()
    if login_clean in users:
        return login_clean
    if "@" in login_clean:
        for key, entry in users.items():
            em = (entry.get("email") or "").strip().lower()
            if em and em == login_clean:
                return key
    return None


def _verify_registered_user_by_key(key: str, password: str) -> bool:
    users = _load_users()
    entry = users.get(key)
    if not entry:
        return False
    salt = entry.get("salt", "")
    stored = entry.get("password_hash", "")
    return _hash_password(password, salt) == stored


def _verify_login(login_identifier: str, password: str) -> bool:
    u = login_identifier.strip()
    if not u:
        return False
    if _verify_demo_login(u, password):
        return True
    key = _get_user_key_for_login(u)
    if not key:
        return False
    return _verify_registered_user_by_key(key, password)


def _username_valid(name: str) -> tuple[bool, str]:
    name = name.strip()
    if len(name) < 3 or len(name) > 32:
        return False, "Username must be 3–32 characters."
    if not re.match(r"^[A-Za-z0-9_]+$", name):
        return False, "Username may only contain letters, numbers, and underscores."
    demo_u, _ = _get_demo_credentials()
    if name.lower() == demo_u.lower():
        return False, f"Username '{demo_u}' is reserved for the demo account."
    return True, ""


def _email_valid(email: str) -> tuple[bool, str]:
    email = email.strip()
    if not email:
        return False, "Email is required."
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return False, "Enter a valid email address."
    return True, ""


def _password_strength(password: str) -> tuple[int, str, str]:
    """Returns (score 0-4, label, color hex)."""
    if not password:
        return 0, "—", "#e2e8f0"
    score = 0
    if len(password) >= 6:
        score += 1
    if len(password) >= 10:
        score += 1
    if re.search(r"[A-Z]", password) and re.search(r"[a-z]", password):
        score += 1
    if re.search(r"\d", password):
        score += 1
    if re.search(r"[^A-Za-z0-9]", password):
        score += 1
    score = min(score, 4)
    labels = ["Weak", "Fair", "Good", "Strong", "Excellent"]
    colors = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#10b981"]
    return score, labels[score], colors[score]


def _email_taken(email: str) -> bool:
    e = email.strip().lower()
    for entry in _load_users().values():
        if (entry.get("email") or "").strip().lower() == e:
            return True
    return False


def _register_user(
    full_name: str,
    email: str,
    username: str,
    password: str,
    confirm: str,
    terms: bool,
) -> tuple[bool, str]:
    if not terms:
        return False, "Please accept the Terms & Conditions to continue."
    fn = full_name.strip()
    if len(fn) < 2:
        return False, "Please enter your full name (at least 2 characters)."
    ok, msg = _email_valid(email)
    if not ok:
        return False, msg
    if _email_taken(email):
        return False, "This email is already registered. Try signing in."
    ok, msg = _username_valid(username)
    if not ok:
        return False, msg
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if password != confirm:
        return False, "Passwords do not match."
    users = _load_users()
    key = username.strip().lower()
    if key in users:
        return False, "That username is already registered. Sign in instead."
    salt = secrets.token_hex(16)
    users[key] = {
        "username": username.strip(),
        "full_name": fn,
        "email": email.strip().lower(),
        "salt": salt,
        "password_hash": _hash_password(password, salt),
    }
    _save_users(users)
    return True, "Account created successfully. Switch to Sign in."


def _inject_auth_styles():
    st.markdown(
        """
        <style>
          /* Auth page: focus layout ss */
          section[data-testid="stSidebar"] { display: none !important; }
          footer { visibility: hidden !important; height: 0 !important; }
          /* Vibrant full-page background (auth screen only) */
          .stApp {
            background: linear-gradient(
              135deg,
              #0c4a6e 0%,
              #1d4ed8 22%,
              #6366f1 45%,
              #7c3aed 68%,
              #c026d3 88%,
              #ec4899 100%
            ) !important;
            background-size: 200% 200% !important;
            animation: auth-page-gradient 18s ease-in-out infinite !important;
          }
          @keyframes auth-page-gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
          }
          .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            z-index: 0;
            pointer-events: none;
            background:
              radial-gradient(ellipse 90% 70% at 15% 20%, rgba(56, 189, 248, 0.45) 0%, transparent 55%),
              radial-gradient(ellipse 80% 60% at 85% 80%, rgba(244, 114, 182, 0.4) 0%, transparent 50%),
              radial-gradient(ellipse 60% 50% at 50% 50%, rgba(167, 139, 250, 0.25) 0%, transparent 60%);
            animation: auth-page-orbs 12s ease-in-out infinite alternate;
          }
          @keyframes auth-page-orbs {
            0% { opacity: 1; transform: scale(1); }
            100% { opacity: 0.92; transform: scale(1.03); }
          }
          [data-testid="stAppViewContainer"] > .main {
            background: transparent !important;
          }
          section.main {
            background: transparent !important;
          }
          .block-container {
            position: relative;
            z-index: 1;
            padding-top: 1.5rem !important;
            max-width: 1200px !important;
          }
          /* Auth-only polish */
          div[data-testid="stVerticalBlock"] > div:has(.auth-card) { background: transparent; }
          .auth-shell {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            border-radius: 28px;
            overflow: hidden;
            box-shadow:
              0 25px 60px rgba(15, 23, 42, 0.35),
              0 0 0 1px rgba(255, 255, 255, 0.15);
            background: rgba(255, 255, 255, 0.06);
            backdrop-filter: blur(8px);
          }
          .auth-hero {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 45%, #a855f7 100%);
            color: #fff;
            padding: 2.5rem 2rem;
            min-height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            position: relative;
            overflow: hidden;
          }
          .auth-hero::before {
            content: '';
            position: absolute;
            width: 320px; height: 320px;
            background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
            top: -80px; right: -80px;
            animation: auth-float 8s ease-in-out infinite;
          }
          .auth-hero::after {
            content: '';
            position: absolute;
            width: 200px; height: 200px;
            background: radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 70%);
            bottom: -40px; left: -40px;
            animation: auth-float 10s ease-in-out infinite reverse;
          }
          @keyframes auth-float {
            0%, 100% { transform: translate(0, 0) scale(1); opacity: 1; }
            50% { transform: translate(12px, -8px) scale(1.05); opacity: 0.85; }
          }
          .auth-hero h2 {
            font-size: 1.75rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin: 0 0 0.5rem 0;
            position: relative;
            z-index: 1;
          }
          .auth-hero p {
            opacity: 0.92;
            font-size: 0.95rem;
            line-height: 1.55;
            margin: 0;
            position: relative;
            z-index: 1;
          }
          .auth-badge {
            display: inline-block;
            background: rgba(255,255,255,0.2);
            backdrop-filter: blur(8px);
            padding: 0.35rem 0.85rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-bottom: 1rem;
            position: relative;
            z-index: 1;
          }
          .auth-card {
            padding: 2rem 1.75rem 2.25rem;
            background: rgba(255, 255, 255, 0.94) !important;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 0 28px 28px 0;
            box-shadow:
              0 25px 50px rgba(15, 23, 42, 0.2),
              inset 0 1px 0 rgba(255, 255, 255, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.35);
          }
          @media (max-width: 900px) {
            .auth-card { border-radius: 0 0 28px 28px; }
          }
          .auth-muted { color: #64748b; font-size: 0.8rem; }
          .auth-link { color: #6366f1 !important; font-weight: 600; text-decoration: none !important; }
          .auth-link:hover { text-decoration: underline !important; }
          .auth-divider {
            display: flex; align-items: center; gap: 0.75rem;
            color: #94a3b8; font-size: 0.8rem; margin: 1rem 0;
          }
          .auth-divider::before, .auth-divider::after {
            content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _inject_dashboard_styles():
    """Vibrant background for the main dashboard (after login)."""
    st.markdown(
        """
        <style>
          .stApp {
            background: linear-gradient(
              160deg,
              #ecfeff 0%,
              #dbeafe 18%,
              #e0e7ff 40%,
              #ede9fe 62%,
              #fae8ff 82%,
              #fce7f3 100%
            ) !important;
            background-size: 220% 220% !important;
            animation: dash-bg-move 25s ease-in-out infinite !important;
          }
          @keyframes dash-bg-move {
            0% { background-position: 0% 40%; }
            50% { background-position: 100% 60%; }
            100% { background-position: 0% 40%; }
          }
          .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            z-index: 0;
            pointer-events: none;
            background:
              radial-gradient(ellipse 70% 55% at 10% 15%, rgba(59, 130, 246, 0.22) 0%, transparent 55%),
              radial-gradient(ellipse 65% 50% at 90% 85%, rgba(168, 85, 247, 0.2) 0%, transparent 52%),
              radial-gradient(ellipse 55% 45% at 50% 50%, rgba(236, 72, 153, 0.1) 0%, transparent 58%);
          }
          [data-testid="stAppViewContainer"] > .main {
            background: transparent !important;
          }
          section.main {
            background: transparent !important;
          }
          section[data-testid="stSidebar"] {
            background: linear-gradient(
              195deg,
              rgba(255, 255, 255, 0.94) 0%,
              rgba(239, 246, 255, 0.9) 50%,
              rgba(250, 245, 255, 0.92) 100%
            ) !important;
            border-right: 1px solid rgba(99, 102, 241, 0.14) !important;
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
          }
          .block-container {
            position: relative;
            z-index: 1;
          }
          section.main div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.72) !important;
            border-radius: 16px !important;
            border: 1px solid rgba(255, 255, 255, 0.85) !important;
            box-shadow: 0 8px 32px rgba(15, 23, 42, 0.06) !important;
            backdrop-filter: blur(8px);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _show_auth_page():
    _inject_auth_styles()
    # Must run before `st.segmented_control(key="auth_segment")` is created (Streamlit rule).
    if st.session_state.pop("auth_pending_signin", False):
        st.session_state["auth_segment"] = "Sign in"

    users = _load_users()
    demo_u, _ = _get_demo_credentials()

    if "auth_view" not in st.session_state:
        st.session_state.auth_view = "signin"

    col_left, col_right = st.columns([1.05, 1], gap="large")

    with col_left:
        st.markdown(
            f"""
            <div class="auth-shell">
              <div class="auth-hero">
                <span class="auth-badge">Climate Intelligence</span>
                <h2>ClimateScope</h2>
                <p style="margin-top:0.75rem;">
                  A modern analytics workspace for global weather trends, risk signals, and AI-assisted planning — built for teams who need clarity, not clutter.
                </p>
                <p style="margin-top:1.25rem;font-size:0.85rem;opacity:0.85;">
                  <strong>{len(users)}</strong> registered locally · Demo: <code style="background:rgba(0,0,0,0.15);padding:2px 8px;border-radius:6px;">{demo_u}</code>
                </p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        flash_ok = st.session_state.pop("auth_flash_success", None)
        if flash_ok:
            st.success(flash_ok)

        seg = st.segmented_control(
            "Navigation",
            options=["Sign in", "Sign up"],
            key="auth_segment",
            label_visibility="collapsed",
        )
        st.session_state.auth_view = "signin" if seg == "Sign in" else "signup"

        st.markdown("##### Welcome back" if seg == "Sign in" else "Create your account")
        st.caption("Sign in to continue" if seg == "Sign in" else "Join ClimateScope in seconds")

        if seg == "Sign in":
            login_id = st.text_input(
                "Email or username",
                key="auth_signin_login",
                placeholder="you@company.com or jane_doe",
                help="Use your registered email or username, or the demo account.",
            )
            show_pw = st.checkbox("Show password", key="auth_signin_show_pw", value=False)
            pw_in = st.text_input(
                "Password",
                type="password" if not show_pw else "default",
                key="auth_signin_pass",
                placeholder="Enter password",
            )

            r1, r2 = st.columns([1, 1])
            with r1:
                remember = st.checkbox("Remember me", key="auth_remember", value=True)
            with r2:
                with st.popover("Forgot password?", use_container_width=True):
                    st.markdown(
                        "Demo build: reset email is not sent. "
                        "Use the **demo account** or **register** a new user. "
                        "For production, connect an email provider or SSO."
                    )

            err_box = st.empty()
            if st.button("Sign in", type="primary", use_container_width=True, key="btn_signin_main"):
                with st.spinner("Signing you in…"):
                    if not login_id.strip():
                        err_box.error("Please enter your email or username.")
                    elif not pw_in:
                        err_box.error("Please enter your password.")
                    elif _verify_login(login_id, pw_in):
                        st.session_state["authenticated"] = True
                        st.session_state["login_user"] = login_id.strip()
                        st.session_state["remember_me"] = remember
                        st.rerun()
                    else:
                        err_box.error("Invalid email/username or password. Check caps lock and try again.")

            st.markdown('<div class="auth-divider">or continue with</div>', unsafe_allow_html=True)
            g1, g2, g3 = st.columns(3)
            with g1:
                if st.button("Google", use_container_width=True, key="soc_google"):
                    st.toast("Google OAuth is not configured in this demo.")
            with g2:
                if st.button("Apple", use_container_width=True, key="soc_apple"):
                    st.toast("Apple Sign In is not configured in this demo.")
            with g3:
                if st.button("Microsoft", use_container_width=True, key="soc_ms"):
                    st.toast("Microsoft OAuth is not configured in this demo.")

            st.markdown(
                '<p class="auth-muted" style="text-align:center;margin-top:1rem;">New here? Use the toggle above to <strong>Sign up</strong>.</p>',
                unsafe_allow_html=True,
            )
        else:
            full_name = st.text_input("Full name", key="auth_full_name", placeholder="Jane Doe")
            email_in = st.text_input("Email", key="auth_email", placeholder="jane@example.com")
            user_in = st.text_input("Username", key="auth_signup_user", placeholder="jane_doe")
            show_sp = st.checkbox("Show passwords", key="auth_signup_show_pw")
            sp = st.text_input(
                "Password",
                type="password" if not show_sp else "default",
                key="auth_signup_pass",
            )
            sc = st.text_input(
                "Confirm password",
                type="password" if not show_sp else "default",
                key="auth_signup_confirm",
            )
            score, label, color = _password_strength(sp)
            st.markdown(
                f"""
                <div style="margin-bottom:0.75rem;">
                  <div style="font-size:0.75rem;color:#64748b;margin-bottom:4px;">Password strength: <strong style="color:{color};">{label}</strong></div>
                  <div style="display:flex;gap:4px;height:6px;border-radius:4px;overflow:hidden;background:#e2e8f0;">
                    <div style="flex:1;background:{color if score >= 1 else '#e2e8f0'};"></div>
                    <div style="flex:1;background:{color if score >= 2 else '#e2e8f0'};"></div>
                    <div style="flex:1;background:{color if score >= 3 else '#e2e8f0'};"></div>
                    <div style="flex:1;background:{color if score >= 4 else '#e2e8f0'};"></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            terms = st.checkbox(
                "I agree to the Terms & Conditions and Privacy Policy",
                key="auth_terms",
                value=False,
            )

            err_signup = st.empty()
            if st.button("Create account", type="primary", use_container_width=True, key="btn_signup_main"):
                with st.spinner("Creating your account…"):
                    ok, message = _register_user(full_name, email_in, user_in, sp, sc, terms)
                    if ok:
                        st.session_state["auth_pending_signin"] = True
                        st.session_state["auth_flash_success"] = message
                        st.rerun()
                    else:
                        err_signup.error(message)

            st.markdown('<div class="auth-divider">or sign up with</div>', unsafe_allow_html=True)
            sg1, sg2, sg3 = st.columns(3)
            with sg1:
                if st.button("Google", use_container_width=True, key="soc_google_up"):
                    st.toast("Google OAuth is not configured in this demo.")
            with sg2:
                if st.button("Apple", use_container_width=True, key="soc_apple_up"):
                    st.toast("Apple Sign In is not configured in this demo.")
            with sg3:
                if st.button("Microsoft", use_container_width=True, key="soc_ms_up"):
                    st.toast("Microsoft OAuth is not configured in this demo.")

            st.markdown(
                '<p class="auth-muted" style="text-align:center;margin-top:1rem;">Already have an account? Switch to <strong>Sign in</strong> above.</p>',
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)


if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    _show_auth_page()
    st.stop()

_inject_dashboard_styles()

st.title("🌍 ClimateScope Global Weather Intelligence System")

with st.expander("📘 How to use this dashboard", expanded=False):
    st.markdown(
        """
        - **Title header**: Shows the ClimateScope Global Weather Intelligence System.
        - **Sidebar**: Contains filters, sliders, and feature menu tabs.
        - **Dynamic analysis panels**: Display interactive charts and maps.
        - **Plotly visualizations**: Support zoom, hover, and tooltips.
        - **Multiple chart types**: Maps, heatmaps, histograms, line graphs, scatter plots, and bar charts.
        - **Real-time updates**: Graphs respond instantly to filter changes.
        - **Dividers**: Separate each analysis module for clarity.
        - **Layout**: Uses Streamlit wide mode for a clean, responsive experience.
        """
    )

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("processed_weather_data.csv")
    df["last_updated"] = pd.to_datetime(df["last_updated"])
    df["year"] = df["last_updated"].dt.year
    df["month"] = df["last_updated"].dt.month
    return df

df = load_data()

temp_col = "temperature_celsius"

# Reset chart export storage on every rerun.
st.session_state["chart_exports"] = []
st.session_state["chart_counter"] = 0
st.session_state["current_feature_heading"] = ""

if not hasattr(st, "_climate_orig_plotly_chart"):
    st._climate_orig_plotly_chart = st.plotly_chart

def _tracked_plotly_chart(fig, *args, **kwargs):
    result = st._climate_orig_plotly_chart(fig, *args, **kwargs)
    try:
        st.session_state["chart_counter"] += 1
        plot_title = None
        if hasattr(fig, "layout") and hasattr(fig.layout, "title") and fig.layout.title:
            plot_title = fig.layout.title.text
        if not plot_title:
            plot_title = f"Plot {st.session_state['chart_counter']:02d}"

        feature_heading = st.session_state.get("current_feature_heading", "").strip()
        if not feature_heading:
            feature_heading = "Dashboard Chart"

        st.session_state["chart_exports"].append(
            {
                "heading": feature_heading,
                "plot_title": str(plot_title),
                "fig": fig,
            }
        )
    except Exception:
        # Do not block UI if export capture fails.
        pass
    return result

st.plotly_chart = _tracked_plotly_chart

# -------------------------------------------------------
# FEATURE FUNCTIONS
# -------------------------------------------------------

def show_key_metrics(df):
    st.subheader("📊 Key Temperature Metrics")
    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average Temp", f"{df[temp_col].mean():.1f}°C")
        col2.metric("Max Temp", f"{df[temp_col].max():.1f}°C")
        col3.metric("Min Temp", f"{df[temp_col].min():.1f}°C")
        col4.metric("Std Dev", f"{df[temp_col].std():.1f}°C")
    else:
        st.write("No data available.")

def hottest_countries(df):
    st.subheader("🔥 Hottest Countries")
    if not df.empty:
        top = df.groupby("country")[temp_col].mean().sort_values(ascending=False).head(10)
        fig = px.bar(x=top.values, y=top.index, orientation="h", labels={"x":"Avg Temp","y":"Country"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No data available.")

def show_distribution(df):
    st.subheader("📊 Temperature Distribution")
    if not df.empty:
        fig = px.histogram(df, x=temp_col, nbins=40)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No data available.")

def seasonal_heatmap(df):
    st.subheader("📅 Seasonal Temperature Heatmap")
    if not df.empty:
        season = df.groupby(["country","month"])[temp_col].mean().reset_index()
        fig = px.density_heatmap(season, x="month", y="country", z=temp_col, color_continuous_scale="Turbo")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No data available.")

def long_term_trend(df):
    st.subheader("📈 Long-term Temperature Trend")
    if not df.empty:
        trend = df.groupby("last_updated")[temp_col].mean().reset_index()
        fig = px.line(trend, x="last_updated", y=temp_col)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No data available.")

def rolling_avg(df):
    st.subheader("📊 Rolling Average Trend")
    if not df.empty:
        window = st.slider("Rolling Window (days)", 3, 30, 7, key="rolling")
        trend = df.groupby("last_updated")[temp_col].mean().reset_index()
        trend["rolling"] = trend[temp_col].rolling(window).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend["last_updated"], y=trend[temp_col], name="Original"))
        fig.add_trace(go.Scatter(x=trend["last_updated"], y=trend["rolling"], name="Rolling Average"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No data available.")

def extreme_events(df):
    st.subheader("⚠ Extreme Weather Events")
    if not df.empty:
        threshold = df[temp_col].quantile(0.95)
        extreme = df[df[temp_col] > threshold]
        if not extreme.empty:
            fig = px.scatter(extreme, x="last_updated", y=temp_col, color="country")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No extreme events detected.")
    else:
        st.write("No data available.")

def similarity(df):
    st.subheader("🌍 Climate Similarity Matrix")
    if not df.empty:
        features = [temp_col]
        if "humidity" in df.columns:
            features.append("humidity")
        if "precip_mm" in df.columns:
            features.append("precip_mm")
        cluster = df.groupby("country")[features].mean()
        if len(cluster) > 1:
            sim = cosine_similarity(cluster)
            fig = px.imshow(sim, x=cluster.index, y=cluster.index, text_auto=True, color_continuous_scale="RdBu_r")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Need more countries for similarity matrix.")
    else:
        st.write("No data available.")

def kruskal_test(df):
    st.subheader("📈 Kruskal-Wallis Test")
    if not df.empty and len(df["country"].unique()) > 1:
        groups = [df[df["country"] == c][temp_col].dropna() for c in df["country"].unique()]
        if all(len(g) > 0 for g in groups):
            stat, p = kruskal(*groups)
            st.write(f"Statistic: {stat:.2f}, p-value: {p:.4f}")
            if p < 0.05:
                st.write("Significant differences between countries.")
            else:
                st.write("No significant differences.")
        else:
            st.write("Insufficient data for test.")
    else:
        st.write("Need multiple countries for Kruskal-Wallis test.")

def multi_param(df):
    st.subheader("🌡️ Multi-parameter Analysis")
    if not df.empty:
        params = [temp_col]
        if "humidity" in df.columns:
            params.append("humidity")
        if "precip_mm" in df.columns:
            params.append("precip_mm")
        if "wind_kph" in df.columns:
            params.append("wind_kph")
        if len(params) > 1:
            fig = px.scatter_matrix(df[params])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Need more parameters.")
    else:
        st.write("No data available.")

def yoy(df):
    st.subheader("📅 Year-over-Year Comparison")
    if not df.empty:
        yoy_data = df.groupby("year")[temp_col].mean().reset_index()
        yoy_data["yoy_change"] = yoy_data[temp_col].pct_change() * 100
        fig = px.bar(yoy_data, x="year", y="yoy_change", labels={"yoy_change":"% Change"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No data available.")

def humidity_wind(df):
    st.subheader("💨 Humidity & Wind Trends")
    if not df.empty and "humidity" in df.columns and "wind_kph" in df.columns:
        trend = df.groupby("last_updated")[["humidity", "wind_kph"]].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend["last_updated"], y=trend["humidity"], name="Humidity"))
        fig.add_trace(go.Scatter(x=trend["last_updated"], y=trend["wind_kph"], name="Wind Speed"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Humidity or wind data not available.")

def dbscan_anomaly(df):
    st.subheader("🔍 DBSCAN Anomaly Detection")
    if not df.empty:
        features = [temp_col]
        if "humidity" in df.columns:
            features.append("humidity")
        data = df[features].dropna()
        if len(data) > 10:
            scaler = StandardScaler()
            X = scaler.fit_transform(data)
            db = DBSCAN(eps=0.5, min_samples=5)
            labels = db.fit_predict(X)
            df_anom = data.copy()
            df_anom["anomaly"] = labels == -1
            fig = px.scatter(df_anom, x=features[0], y=features[1] if len(features)>1 else features[0], color="anomaly")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Insufficient data for anomaly detection.")
    else:
        st.write("No data available.")

def monthly_dist(df):
    st.subheader("📅 Monthly Temperature Distribution")
    if not df.empty:
        fig = px.box(df, x="month", y=temp_col)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No data available.")

def percentile(df):
    st.subheader("📊 Temperature Percentiles")
    if not df.empty:
        perc = df[temp_col].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        st.write(perc)
        fig = px.line(x=perc.index, y=perc.values, markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No data available.")

def conditions(df):
    st.subheader("🌤️ Weather Conditions Distribution")
    if not df.empty and "condition" in df.columns:
        cond = df["condition"].value_counts()
        fig = px.pie(values=cond.values, names=cond.index)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Weather condition data not available.")

def forecast(df):
    st.subheader("🔮 Temperature Forecast")
    if not df.empty:
        year_temp = df.groupby("year")[temp_col].mean().reset_index()
        if len(year_temp) > 1:
            X = year_temp[["year"]]
            y = year_temp[temp_col]
            model = LinearRegression()
            model.fit(X, y)
            future = pd.DataFrame({"year": np.arange(year_temp["year"].max()+1, year_temp["year"].max()+6)})
            future["prediction"] = model.predict(future)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=year_temp["year"], y=year_temp[temp_col], mode="lines+markers", name="Historical"))
            fig.add_trace(go.Scatter(x=future["year"], y=future["prediction"], mode="lines+markers", name="Forecast"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Insufficient data for forecasting.")
    else:
        st.write("No data available.")

def corr_vars(df):
    st.subheader("🔗 Correlation Analysis")
    if not df.empty:
        numeric = df.select_dtypes(include=np.number)
        if not numeric.empty:
            corr = numeric.corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No numeric data.")
    else:
        st.write("No data available.")

def precipitation_rainfall(df):
    st.subheader("🌧 Precipitation Analysis")
    if not df.empty and "precip_mm" in df.columns:
        rain = df.groupby("country")["precip_mm"].mean().sort_values(ascending=False).head(10)
        fig = px.bar(x=rain.values, y=rain.index, orientation="h", labels={"x":"Avg Precip","y":"Country"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Precipitation data not available.")

def hottest_on_globe(df):
    # Already covered in hottest_countries
    pass

def latitude_longitude_map(df):
    st.subheader("🌍 Lat-Long Temperature Map")
    if not df.empty and "latitude" in df.columns and "longitude" in df.columns:
        fig = px.scatter_geo(df, lat="latitude", lon="longitude", color=temp_col, hover_name="country", color_continuous_scale="Turbo")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Location data not available.")

def top_countries_comparison(df):
    st.subheader("🏆 Top Countries Comparison")
    if not df.empty:
        top_countries = df.groupby("country")[temp_col].mean().sort_values(ascending=False).head(10).index
        comp = df[df["country"].isin(top_countries)].groupby("country")[temp_col].agg(["mean", "max", "min"]).reset_index()
        st.dataframe(comp)
        fig = px.bar(comp, x="country", y="mean", title="Average Temperature")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No data available.")

def global_insight_summary_panel(df):
    st.subheader("🌐 Global Insights")
    if not df.empty:
        st.write(f"Total Countries: {df['country'].nunique()}")
        st.write(f"Average Global Temp: {df[temp_col].mean():.1f}°C")
        st.write(f"Data Points: {len(df)}")
    else:
        st.write("No data available.")

def interactive_feature_explanations(df):
    st.subheader("ℹ️ Feature Explanations")
    st.write("This dashboard provides various analyses of global weather data.")
    st.write("- **Key Metrics**: Basic statistics of temperature.")
    st.write("- **Maps**: Visualize temperature geographically.")
    # Add more explanations as needed

def time_series_decomposition(df):
    st.subheader("📉 Time Series Decomposition")
    if not df.empty:
        ts = df.groupby("last_updated")[temp_col].mean()
        ts = ts.asfreq("D").ffill()
        if len(ts) > 60:
            result = seasonal_decompose(ts, model="additive", period=30)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts.index, y=result.trend, name="Trend"))
            fig.add_trace(go.Scatter(x=ts.index, y=result.seasonal, name="Seasonal"))
            fig.add_trace(go.Scatter(x=ts.index, y=result.resid, name="Residual"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Insufficient data.")
    else:
        st.write("No data available.")

def heatwave_detection(df):
    st.subheader("🔥 Heatwave Detection")
    if not df.empty:
        heat_threshold = df[temp_col].quantile(0.90)
        heatwaves = df[df[temp_col] > heat_threshold]
        st.metric("Heatwave Events", len(heatwaves))
        if not heatwaves.empty:
            fig = px.scatter(heatwaves, x="last_updated", y=temp_col, color="country")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No heatwaves detected.")
    else:
        st.write("No data available.")

def seasonal_rainfall_comparison(df):
    st.subheader("🌧 Seasonal Rainfall Comparison")
    if not df.empty and "precip_mm" in df.columns:
        rain = df.groupby(["country", "month"])["precip_mm"].mean().reset_index()
        fig = px.line(rain, x="month", y="precip_mm", color="country")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Rainfall data not available.")

def climate_clustering(df):
    st.subheader("🌎 Climate Zone Clustering (K-Means)")
    if not df.empty:
        features = [temp_col]
        if "humidity" in df.columns:
            features.append("humidity")
        if "precip_mm" in df.columns:
            features.append("precip_mm")

        cluster_data = df.groupby("country")[features].mean().dropna()

        if len(cluster_data) > 3:
            scaler = StandardScaler()
            X = scaler.fit_transform(cluster_data)

            kmeans = KMeans(n_clusters=3, random_state=42)
            cluster_data["cluster"] = kmeans.fit_predict(X)

            fig = px.scatter(
                cluster_data,
                x=features[0],
                y=features[1] if len(features) > 1 else features[0],
                color="cluster",
                hover_name=cluster_data.index,
                title="Climate Zone Clustering"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Not enough countries for clustering.")
    else:
        st.write("No data available.")

def climate_risk_index(df):
    st.subheader("⚠ Climate Risk Index")
    if not df.empty:
        agg_dict = {temp_col: "mean"}
        if "precip_mm" in df.columns:
            agg_dict["precip_mm"] = "mean"

        risk = df.groupby("country").agg(agg_dict).reset_index()
        risk["risk_score"] = risk[temp_col].rank(pct=True)

        top_risk = risk.sort_values("risk_score", ascending=False).head(10)
        fig = px.bar(
            top_risk,
            x="risk_score",
            y="country",
            orientation="h",
            title="Top Climate Risk Countries"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No data available.")

def climate_change_indicator(df):
    st.subheader("🌡 Climate Change Indicator")
    if not df.empty:
        trend = df.groupby("year")[temp_col].mean().reset_index()
        if len(trend) > 1:
            X = trend[["year"]]
            y = trend[temp_col]

            model = LinearRegression()
            model.fit(X, y)

            slope = model.coef_[0]
            st.metric("Temperature Trend per Year", f"{slope:.3f} °C/year")

            fig = px.scatter(trend, x="year", y=temp_col, trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Insufficient data for climate change indicator.")
    else:
        st.write("No data available.")

def _temp_range_by_preference(preference):
    if preference == "Cool":
        return (18.0, 24.0)
    if preference == "Warm":
        return (25.0, 32.0)
    return (22.0, 28.0)

def _event_weights(event_type):
    weights = {
        "Outdoor Wedding": {"heat": 0.30, "rain": 0.30, "wind": 0.10, "humidity": 0.20, "volatility": 0.10},
        "College Fest": {"heat": 0.28, "rain": 0.22, "wind": 0.12, "humidity": 0.20, "volatility": 0.18},
        "Trek/Adventure": {"heat": 0.32, "rain": 0.18, "wind": 0.22, "humidity": 0.10, "volatility": 0.18},
        "Beach Trip": {"heat": 0.18, "rain": 0.34, "wind": 0.16, "humidity": 0.22, "volatility": 0.10},
        "Sports Event": {"heat": 0.30, "rain": 0.24, "wind": 0.18, "humidity": 0.14, "volatility": 0.14},
    }
    return weights[event_type]

def _safe_minmax(series):
    if series.empty:
        return series
    s_min = series.min()
    s_max = series.max()
    if s_max == s_min:
        return pd.Series(0.0, index=series.index)
    return (series - s_min) / (s_max - s_min)

def _build_event_scores(df, event_type, comfort_pref, preferred_months):
    if df.empty:
        return pd.DataFrame()
    scoped = df[df["month"].isin(preferred_months)].copy()
    if scoped.empty:
        return pd.DataFrame()

    agg_cols = {temp_col: "mean"}
    if "humidity" in scoped.columns:
        agg_cols["humidity"] = "mean"
    if "precip_mm" in scoped.columns:
        agg_cols["precip_mm"] = "mean"
    if "wind_kph" in scoped.columns:
        agg_cols["wind_kph"] = "mean"

    score_df = scoped.groupby(["country", "month"]).agg(agg_cols).reset_index()
    volatility = scoped.groupby(["country", "month"])[temp_col].std().reset_index(name="temp_std")
    score_df = score_df.merge(volatility, on=["country", "month"], how="left")
    score_df["temp_std"] = score_df["temp_std"].fillna(0.0)

    low, high = _temp_range_by_preference(comfort_pref)
    temp_deviation = np.where(
        score_df[temp_col] < low, low - score_df[temp_col],
        np.where(score_df[temp_col] > high, score_df[temp_col] - high, 0.0)
    )
    score_df["heat_risk"] = _safe_minmax(pd.Series(temp_deviation, index=score_df.index))
    score_df["rain_risk"] = _safe_minmax(score_df["precip_mm"]) if "precip_mm" in score_df.columns else 0.0
    score_df["wind_risk"] = _safe_minmax(score_df["wind_kph"]) if "wind_kph" in score_df.columns else 0.0
    score_df["humidity_risk"] = _safe_minmax((score_df["humidity"] - 55).abs()) if "humidity" in score_df.columns else 0.0
    score_df["volatility_risk"] = _safe_minmax(score_df["temp_std"])

    w = _event_weights(event_type)
    score_df["risk_score"] = (
        w["heat"] * score_df["heat_risk"]
        + w["rain"] * score_df["rain_risk"]
        + w["wind"] * score_df["wind_risk"]
        + w["humidity"] * score_df["humidity_risk"]
        + w["volatility"] * score_df["volatility_risk"]
    )
    score_df["suitability_score"] = (100 - score_df["risk_score"] * 100).round(1)
    return score_df.sort_values("suitability_score", ascending=False)

def event_planner_recommendations(df, event_type, comfort_pref, preferred_months):
    st.subheader("🎯 Event Planner Recommendations")
    scored = _build_event_scores(df, event_type, comfort_pref, preferred_months)
    if scored.empty:
        st.write("No recommendation data available for selected filters.")
        return
    top = scored.head(10)[["country", "month", "suitability_score", "risk_score"]]
    st.dataframe(top, use_container_width=True)
    fig = px.bar(top.sort_values("suitability_score"), x="suitability_score", y="country", orientation="h", color="month")
    st.plotly_chart(fig, use_container_width=True)

def event_planner_heatmap(df, event_type, comfort_pref, preferred_months):
    st.subheader("🗓 Event Suitability Heatmap")
    scored = _build_event_scores(df, event_type, comfort_pref, preferred_months)
    if scored.empty:
        st.write("No data available for heatmap.")
        return
    heat = scored.pivot_table(index="country", columns="month", values="suitability_score", aggfunc="mean")
    fig = px.imshow(heat, aspect="auto", color_continuous_scale="Turbo")
    st.plotly_chart(fig, use_container_width=True)

def event_planner_map(df, event_type, comfort_pref, preferred_months):
    st.subheader("🗺 Best Event Locations Map")
    scored = _build_event_scores(df, event_type, comfort_pref, preferred_months)
    if scored.empty:
        st.write("No mapping data available.")
        return

    best_countries = scored.groupby("country")["suitability_score"].max().reset_index().sort_values("suitability_score", ascending=False).head(15)
    if "latitude" in df.columns and "longitude" in df.columns:
        loc = df.groupby("country")[["latitude", "longitude"]].mean().reset_index()
        mapped = best_countries.merge(loc, on="country", how="left").dropna(subset=["latitude", "longitude"])
        if mapped.empty:
            st.write("Latitude/longitude data unavailable for selected countries.")
            return
        fig = px.scatter_geo(
            mapped,
            lat="latitude",
            lon="longitude",
            color="suitability_score",
            size="suitability_score",
            hover_name="country",
            color_continuous_scale="Turbo",
        )
    else:
        fig = px.choropleth(
            best_countries,
            locations="country",
            locationmode="country names",
            color="suitability_score",
            color_continuous_scale="Turbo",
        )
    st.plotly_chart(fig, use_container_width=True)

def _charts_pdf_bytes():
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    page_width, page_height = letter

    charts = st.session_state.get("chart_exports", [])
    for idx, item in enumerate(charts, start=1):
        heading = item.get("heading", f"Chart {idx}")
        plot_title = item.get("plot_title", f"Plot {idx}")
        fig = item.get("fig")

        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(40, page_height - 40, f"Graph: {heading}")
        pdf.setFont("Helvetica", 11)
        pdf.drawString(40, page_height - 60, f"Plot Title: {plot_title}")

        # Convert Plotly figure to PNG for embedding in PDF.
        image_bytes = fig.to_image(format="png", width=1400, height=800, scale=1)
        image_reader = ImageReader(BytesIO(image_bytes))
        img_w, img_h = image_reader.getSize()

        max_w = page_width - 80
        max_h = page_height - 140
        scale = min(max_w / img_w, max_h / img_h)
        draw_w = img_w * scale
        draw_h = img_h * scale

        x = (page_width - draw_w) / 2
        y = 40
        pdf.drawImage(image_reader, x, y, width=draw_w, height=draw_h, preserveAspectRatio=True, mask="auto")
        pdf.showPage()

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()

def data_export(df):
    st.subheader("⬇ Download Charts Report (Single PDF)")
    chart_count = len(st.session_state.get("chart_exports", []))
    if chart_count == 0:
        st.write("No charts available yet. Enable features and render charts first.")
        return
    try:
        pdf_bytes = _charts_pdf_bytes()
    except Exception as exc:
        st.error(
            "Unable to generate chart PDF. Install image export dependency with: "
            "`pip install kaleido` and restart Streamlit."
        )
        st.caption(f"Export error: {exc}")
        return
    st.download_button(
        f"Download All Charts ({chart_count}) as PDF",
        data=pdf_bytes,
        file_name="dashboard_charts_report.pdf",
        mime="application/pdf",
    )

# Global choropleth map
def global_choropleth_map(df):
    st.subheader("🌍 Global Temperature Choropleth")
    if not df.empty:
        # Need to map country names to ISO codes
        country_temp = df.groupby("country")[temp_col].mean().reset_index()
        # Assuming country names are standard, use plotly's built-in
        fig = px.choropleth(country_temp, locations="country", locationmode="country names", color=temp_col, color_continuous_scale="Turbo")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No data available.")

# -------------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------------

st.sidebar.header("🔎 Filters")
if st.sidebar.button("Log out", use_container_width=True):
    st.session_state["authenticated"] = False
    for k in ("login_user", "chart_exports", "chart_counter", "current_feature_heading"):
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()
if st.session_state.get("login_user"):
    st.sidebar.caption(f"Signed in as **{st.session_state['login_user']}**")

country_options = sorted(df["country"].dropna().unique())

# default to India + first other available country (if present)
default_countries = []
if "India" in country_options:
    default_countries.append("India")
if country_options:
    first_other = next((c for c in country_options if c != "India"), None)
    if first_other:
        default_countries.append(first_other)

countries = st.sidebar.multiselect(
    "Select Countries",
    country_options,
    default=default_countries
)

years = st.sidebar.slider(
    "Select Year Range",
    int(df["year"].min()),
    int(df["year"].max()),
    (int(df["year"].min()), int(df["year"].max()))
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Event Planner Controls")
event_type = st.sidebar.selectbox(
    "Event Type",
    ["Outdoor Wedding", "College Fest", "Trek/Adventure", "Beach Trip", "Sports Event"],
    index=1,
)
comfort_pref = st.sidebar.selectbox("Comfort Preference", ["Cool", "Moderate", "Warm"], index=1)
preferred_months = st.sidebar.multiselect("Preferred Months", options=list(range(1, 13)), default=[11, 12, 1, 2])
if not preferred_months:
    preferred_months = list(range(1, 13))

filtered_df = df[
    (df["country"].isin(countries)) &
    (df["year"].between(years[0], years[1]))
]

# -------------------------------------------------------
# FEATURE GROUPS
# -------------------------------------------------------

feature_groups = {
    "Overview": {
        "Key Metrics": show_key_metrics,
        "Hottest Countries": hottest_countries,
        "Top Countries Comparison": top_countries_comparison,
        "Global Insights": global_insight_summary_panel,
        "Feature Explanations": interactive_feature_explanations,
    },
    "Trends & Patterns": {
        "Long-term Trend": long_term_trend,
        "Rolling Average": rolling_avg,
        "Year-over-Year": yoy,
        "Time Series Decomposition": time_series_decomposition,
        "Seasonal Rainfall": seasonal_rainfall_comparison,
    },
    "Anomalies": {
        "Extreme Events": extreme_events,
        "Heatwave Detection": heatwave_detection,
        "DBSCAN Anomalies": dbscan_anomaly,
    },
    "Distribution": {
        "Temperature Distribution": show_distribution,
        "Seasonal Heatmap": seasonal_heatmap,
        "Monthly Distribution": monthly_dist,
        "Percentile Analysis": percentile,
        "Weather Conditions": conditions,
    },
    "Forecasting & Correlation": {
        "Temperature Forecast": forecast,
        "Correlation Heatmap": corr_vars,
        "Similarity Matrix": similarity,
        "Kruskal-Wallis Test": kruskal_test,
        "Climate Change Indicator": climate_change_indicator,
        "Climate Risk Index": climate_risk_index,
        "Climate Clustering": climate_clustering,
    },
    "Maps": {
        "Global Choropleth": global_choropleth_map,
        "Lat-Long Scatter": latitude_longitude_map,
    },
    "Other": {
        "Precipitation Analysis": precipitation_rainfall,
        "Multi-parameter": multi_param,
        "Humidity & Wind": humidity_wind,
    },
    "Event Planner": {
        "Recommendations": lambda d: event_planner_recommendations(d, event_type, comfort_pref, preferred_months),
        "Suitability Heatmap": lambda d: event_planner_heatmap(d, event_type, comfort_pref, preferred_months),
        "Best Locations Map": lambda d: event_planner_map(d, event_type, comfort_pref, preferred_months),
    }
}

# -------------------------------------------------------
# MAIN INTERFACE
# -------------------------------------------------------

tabs = st.tabs(list(feature_groups.keys()))

for tab, (tab_name, features) in zip(tabs, feature_groups.items()):
    with tab:
        st.header(f"📊 {tab_name}")
        selected_features = []
        for feature_name in features.keys():
            if st.checkbox(feature_name, key=f"{tab_name}_{feature_name}", value=True):
                selected_features.append(feature_name)
        
        for feature in selected_features:
            st.session_state["current_feature_heading"] = f"{tab_name} - {feature}"
            features[feature](filtered_df)
            st.markdown("---")

st.markdown("### ⬇ Download All Charts")
data_export(filtered_df)

st.success("✅ ClimateScope Global Weather Intelligence System dashboard is ready. Adjust the sidebar filters to explore global weather insights in real time.")