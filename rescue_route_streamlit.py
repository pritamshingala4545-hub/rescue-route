"""
RescueRoute - Smart Emergency Ambulance Dispatch & Route Optimizer
A complete, self-contained Streamlit + SQLite emergency management platform.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import math
import random
import sqlite3
import datetime as dt
from contextlib import contextmanager

import pandas as pd
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.express as px

# ============================================================================
# 1. CONFIG
# ============================================================================

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rescue_route.db")

AHMEDABAD_CENTER = (23.0225, 72.5714)

EMERGENCY_TYPES = ["Accident", "Heart Attack", "Stroke", "Breathing Problem",
                    "Trauma", "Pregnancy", "Other"]
SEVERITY_LEVELS = ["Medium", "High", "Critical"]
SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2}
AMBULANCE_TYPES = ["Basic Life Support", "Advanced Life Support", "ICU on Wheels", "Neonatal"]
STATUS_FLOW = ["Requested", "Dispatched", "On the Way", "At Patient",
               "Hospital Selected", "Completed", "Cancelled"]
TRAFFIC_LEVELS = ["Low", "Moderate", "Heavy"]

st.set_page_config(
    page_title="RescueRoute | Emergency Dispatch",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# 2. THEME / CSS
# ============================================================================

def inject_css():
    st.markdown("""
    <style>
    :root{
        --navy:#0b1220; --navy2:#111a2e; --red:#e63946; --red-dark:#c1121f;
        --text:#eef2f7; --muted:#9fb0c7; --card:#141f38; --border:#22314f;
        --green:#2ecc71; --orange:#f5a623; --gray:#7d8ba1;
    }
    .stApp{ background: radial-gradient(1200px 600px at 10% -10%, #16213b 0%, var(--navy) 55%) fixed; }
    section[data-testid="stSidebar"]{
        background: linear-gradient(180deg, var(--navy2) 0%, var(--navy) 100%);
        border-right: 1px solid var(--border);
    }
    h1, h2, h3, h4, .stMarkdown, label, p, span, div { color: var(--text); }
    .rr-hero{
        background: linear-gradient(120deg, #7f1d1d 0%, var(--red-dark) 45%, #7a1120 100%);
        border-radius: 18px; padding: 28px 32px; margin-bottom: 18px;
        box-shadow: 0 12px 30px rgba(198,30,42,0.25);
        border: 1px solid rgba(255,255,255,0.08);
    }
    .rr-hero h1{ margin:0; font-size:2rem; letter-spacing:.5px;}
    .rr-hero p{ margin:6px 0 0 0; color:#ffe3e6; opacity:.9;}
    .rr-card{
        background: var(--card); border:1px solid var(--border); border-radius:14px;
        padding:18px 20px; box-shadow: 0 6px 18px rgba(0,0,0,0.25); margin-bottom:14px;
    }
    .rr-kpi{
        background: linear-gradient(160deg, var(--card) 0%, #101a30 100%);
        border:1px solid var(--border); border-radius:16px; padding:18px;
        text-align:center; box-shadow: 0 6px 16px rgba(0,0,0,0.3);
    }
    .rr-kpi .val{ font-size:2rem; font-weight:800; color:#fff; }
    .rr-kpi .lbl{ color: var(--muted); font-size:.85rem; text-transform:uppercase; letter-spacing:.6px;}
    .badge{ padding:4px 12px; border-radius:999px; font-size:.78rem; font-weight:700; display:inline-block;}
    .b-green{ background: rgba(46,204,113,.18); color:#4ee08a; border:1px solid rgba(46,204,113,.4);}
    .b-red{ background: rgba(230,57,70,.18); color:#ff7b85; border:1px solid rgba(230,57,70,.4);}
    .b-orange{ background: rgba(245,166,35,.18); color:#ffc266; border:1px solid rgba(245,166,35,.4);}
    .b-gray{ background: rgba(125,139,161,.18); color:#c2cee0; border:1px solid rgba(125,139,161,.4);}
    .b-blue{ background: rgba(66,135,245,.18); color:#8fb8ff; border:1px solid rgba(66,135,245,.4);}
    .stButton>button{
        background: linear-gradient(120deg, var(--red) 0%, var(--red-dark) 100%);
        color:white; border:none; border-radius:10px; padding:.5rem 1.1rem; font-weight:700;
        box-shadow: 0 4px 12px rgba(198,30,42,.35);
    }
    .stButton>button:hover{ filter: brightness(1.08); }
    .stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"]{
        background: var(--card) !important; color: var(--text) !important; border-radius:8px !important;
        border: 1px solid var(--border) !important;
    }
    div[data-testid="stMetric"]{
        background: var(--card); border:1px solid var(--border); border-radius:14px; padding:12px;
    }
    hr{ border-color: var(--border); }
    .login-wrap{ max-width:460px; margin: 40px auto; }
    </style>
    """, unsafe_allow_html=True)


def badge(text, kind="gray"):
    return f'<span class="badge b-{kind}">{text}</span>'


STATUS_BADGE = {
    "Available": "green", "Busy": "red", "Maintenance": "orange", "Unavailable": "gray",
    "Requested": "blue", "Dispatched": "orange", "On the Way": "orange",
    "At Patient": "blue", "Hospital Selected": "blue", "Completed": "green", "Cancelled": "gray",
}

# ============================================================================
# 3. DATABASE LAYER
# ============================================================================

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def execute(query, params=()):
    """Run an INSERT/UPDATE/DELETE. Returns lastrowid."""
    with get_conn() as conn:
        cur = conn.execute(query, params)
        return cur.lastrowid


def read(query, params=()):
    """Run a SELECT and return a pandas DataFrame (always fresh, no caching)."""
    with get_conn() as conn:
        try:
            df = pd.read_sql_query(query, conn, params=params)
        except Exception:
            df = pd.DataFrame()
        return df


def read_one(query, params=()):
    with get_conn() as conn:
        cur = conn.execute(query, params)
        row = cur.fetchone()
        return dict(row) if row else None


def now_str():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def init_db():
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            name TEXT,
            phone TEXT,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS ambulances(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ambulance_id TEXT UNIQUE NOT NULL,
            driver TEXT NOT NULL,
            driver_email TEXT UNIQUE NOT NULL,
            phone TEXT,
            ambulance_type TEXT,
            latitude REAL,
            longitude REAL,
            status TEXT DEFAULT 'Available',
            created_at TEXT,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS hospitals(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            address TEXT,
            contact TEXT,
            latitude REAL,
            longitude REAL,
            emergency_available INTEGER DEFAULT 1,
            icu_available INTEGER DEFAULT 1,
            beds INTEGER DEFAULT 0,
            created_at TEXT,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS emergencies(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT,
            patient_name TEXT,
            patient_phone TEXT,
            emergency_type TEXT,
            severity TEXT,
            patient_lat REAL,
            patient_lon REAL,
            ambulance_id TEXT,
            hospital_id INTEGER,
            status TEXT DEFAULT 'Requested',
            traffic TEXT,
            ambulance_distance REAL,
            eta REAL,
            hospital_distance REAL,
            created_at TEXT,
            updated_at TEXT,
            completed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS location_updates(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            emergency_id INTEGER,
            ambulance_id TEXT,
            latitude REAL,
            longitude REAL,
            updated_at TEXT
        );
        """)
    seed_demo_data()


def seed_demo_data():
    users_count = read("SELECT COUNT(*) c FROM users")["c"].iloc[0]
    if users_count == 0:
        demo_users = [
            ("admin@rescue.com", "admin123", "Admin", "System Admin", "9999900000"),
            ("operator@rescue.com", "operator123", "Operator", "Dispatch Operator", "9999900001"),
            ("user@rescue.com", "user123", "User", "Demo Patient", "9999900002"),
            ("rajesh@rescue.com", "driver123", "Driver", "Rajesh Patel", "9999900011"),
            ("amit@rescue.com", "driver123", "Driver", "Amit Shah", "9999900012"),
            ("vijay@rescue.com", "driver123", "Driver", "Vijay Solanki", "9999900013"),
            ("karan@rescue.com", "driver123", "Driver", "Karan Desai", "9999900014"),
            ("nilesh@rescue.com", "driver123", "Driver", "Nilesh Joshi", "9999900015"),
        ]
        for email, pwd, role, name, phone in demo_users:
            execute(
                "INSERT OR IGNORE INTO users(email,password,role,name,phone,created_at) VALUES(?,?,?,?,?,?)",
                (email, pwd, role, name, phone, now_str()),
            )

    amb_count = read("SELECT COUNT(*) c FROM ambulances")["c"].iloc[0]
    if amb_count == 0:
        demo_ambulances = [
            ("AMB-101", "Rajesh Patel", "rajesh@rescue.com", "9999900011", "Advanced Life Support", 23.0225, 72.5714),
            ("AMB-102", "Amit Shah", "amit@rescue.com", "9999900012", "Basic Life Support", 23.0300, 72.5800),
            ("AMB-103", "Vijay Solanki", "vijay@rescue.com", "9999900013", "ICU on Wheels", 23.0100, 72.5600),
            ("AMB-104", "Karan Desai", "karan@rescue.com", "9999900014", "Basic Life Support", 22.9950, 72.5500),
            ("AMB-105", "Nilesh Joshi", "nilesh@rescue.com", "9999900015", "Advanced Life Support", 23.0400, 72.5900),
        ]
        for amb_id, driver, email, phone, atype, lat, lon in demo_ambulances:
            execute(
                """INSERT OR IGNORE INTO ambulances
                (ambulance_id, driver, driver_email, phone, ambulance_type, latitude, longitude,
                 status, created_at, updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (amb_id, driver, email, phone, atype, lat, lon, "Available", now_str(), now_str()),
            )

    hosp_count = read("SELECT COUNT(*) c FROM hospitals")["c"].iloc[0]
    if hosp_count == 0:
        demo_hospitals = [
            ("Civil Hospital Ahmedabad", "Asarwa, Ahmedabad", "079-22680000", 23.0432, 72.6023, 1, 1, 120),
            ("Sterling Hospital", "Gurukul Road, Ahmedabad", "079-40011000", 23.0448, 72.5270, 1, 1, 90),
            ("SAL Hospital", "Drive-in Road, Ahmedabad", "079-71700100", 23.0530, 72.5150, 1, 1, 60),
            ("Zydus Hospital", "Thaltej, Ahmedabad", "079-66120000", 23.0490, 72.5050, 1, 1, 80),
            ("Shalby Hospital", "S.G. Highway, Ahmedabad", "079-40203040", 23.0330, 72.5150, 1, 0, 50),
            ("Apollo Hospital", "Bhat, Gandhinagar", "079-66701800", 23.1200, 72.5900, 1, 1, 100),
        ]
        for name, addr, contact, lat, lon, ea, icu, beds in demo_hospitals:
            execute(
                """INSERT OR IGNORE INTO hospitals
                (name, address, contact, latitude, longitude, emergency_available, icu_available,
                 beds, created_at, updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (name, addr, contact, lat, lon, ea, icu, beds, now_str(), now_str()),
            )


# ============================================================================
# 4. UTILITIES
# ============================================================================

def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in KM."""
    try:
        lat1, lon1, lat2, lon2 = map(float, (lat1, lon1, lat2, lon2))
    except (TypeError, ValueError):
        return float("inf")
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def auto_ambulance_id():
    df = read("SELECT ambulance_id FROM ambulances")
    max_num = 100
    for aid in df["ambulance_id"].tolist():
        try:
            num = int(str(aid).split("-")[1])
            max_num = max(max_num, num)
        except (IndexError, ValueError):
            continue
    return f"AMB-{max_num + 1}"


def nearest_available_ambulance(lat, lon):
    df = read("SELECT * FROM ambulances WHERE status='Available'")
    if df.empty:
        return None
    df["dist"] = df.apply(lambda r: haversine(lat, lon, r["latitude"], r["longitude"]), axis=1)
    df = df.sort_values("dist")
    return df.iloc[0].to_dict()


def estimate_traffic():
    return random.choice(TRAFFIC_LEVELS)


def estimate_eta(distance_km, traffic="Moderate"):
    speed_map = {"Low": 45, "Moderate": 32, "Heavy": 20}
    speed = speed_map.get(traffic, 30)
    minutes = (distance_km / max(speed, 1)) * 60
    return round(max(minutes, 2), 1)


def nearby_hospitals(lat, lon, radius_km=5.0):
    df = read("SELECT * FROM hospitals")
    if df.empty:
        return df
    df["dist"] = df.apply(lambda r: haversine(lat, lon, r["latitude"], r["longitude"]), axis=1)
    df = df[df["dist"] <= radius_km].sort_values("dist")
    return df


OSRM_BASE_URL = "https://router.project-osrm.org/route/v1/driving"


def get_real_route(start_lat, start_lon, end_lat, end_lon):
    """
    Fetch a REAL road route between two points using the public OSRM routing API
    (OpenStreetMap road network). No simulated/straight-line/jittered data.

    OSRM expects coordinates as "longitude,latitude" — the opposite order of our
    database's "latitude,longitude" convention — so that reordering happens here.

    Returns a dict on success:
        {
            "coordinates": [[lat, lon], ...],  # ready for folium.PolyLine (lat,lon order)
            "distance_km": float,               # actual road distance from OSRM
            "duration_min": float,              # actual estimated duration from OSRM
        }
    Returns None if the routing service is unavailable, times out, or returns no
    route — callers must handle this by showing a clear "service unavailable"
    message rather than drawing a fake route.
    """
    url = (
        f"{OSRM_BASE_URL}/{start_lon},{start_lat};{end_lon},{end_lat}"
        f"?overview=full&geometries=geojson&steps=true"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "Ok" or not data.get("routes"):
            return None
        route = data["routes"][0]
        geometry = route["geometry"]["coordinates"]  # OSRM returns [lon, lat] pairs
        coordinates = [[lat, lon] for lon, lat in geometry]
        if not coordinates:
            return None
        return {
            "coordinates": coordinates,
            "distance_km": round(route["distance"] / 1000.0, 2),
            "duration_min": round(route["duration"] / 60.0, 1),
        }
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
        return None


def route_options(lat1, lon1, lat2, lon2):
    """
    Real road route option(s) between two points, sourced entirely from OSRM
    (distance and duration are the actual routing-engine values — no fake
    multipliers, no random traffic). Returns an empty list if the routing
    service is unavailable so callers can show a clear warning instead of a
    fake route.
    """
    route = get_real_route(lat1, lon1, lat2, lon2)
    if route is None:
        return []
    return [{
        "route": "Real Road Route (OSRM)",
        "distance": route["distance_km"],
        "duration": route["duration_min"],
        "coordinates": route["coordinates"],
        "recommended": True,
    }]


def jitter_path(lat1, lon1, lat2, lon2, seed=0):
    """Produce a simple curved polyline between two points for map display (simulation)."""
    rnd = random.Random(seed)
    pts = []
    steps = 6
    for i in range(steps + 1):
        t = i / steps
        lat = lat1 + (lat2 - lat1) * t
        lon = lon1 + (lon2 - lon1) * t
        if 0 < i < steps:
            lat += rnd.uniform(-0.003, 0.003)
            lon += rnd.uniform(-0.003, 0.003)
        pts.append((lat, lon))
    return pts


def safe_rerun():
    st.rerun()


def flash(msg, kind="success"):
    st.session_state["_flash"] = (kind, msg)


def show_flash():
    f = st.session_state.pop("_flash", None)
    if f:
        kind, msg = f
        getattr(st, kind)(msg)


# ============================================================================
# 5. AUTH
# ============================================================================

def authenticate(email, password):
    row = read_one("SELECT * FROM users WHERE email=? AND password=?", (email.strip().lower(), password))
    return row


def login_page():
    inject_css()
    st.markdown("""
    <div class="rr-hero" style="text-align:center;">
        <h1>🚑 RescueRoute</h1>
        <p>Smart Emergency Ambulance Dispatch & Route Optimizer</p>
    </div>
    """, unsafe_allow_html=True)

    col = st.columns([1, 1.2, 1])[1]
    with col:
        st.markdown('<div class="rr-card">', unsafe_allow_html=True)
        st.subheader("Sign in")
        email = st.text_input("Email", placeholder="you@rescue.com")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        if st.button("Login", use_container_width=True):
            if not email or not password:
                st.error("Please enter both email and password.")
            else:
                user = authenticate(email, password)
                if user:
                    st.session_state["user"] = user
                    flash(f"Welcome back, {user['name']}!")
                    safe_rerun()
                else:
                    st.error("Invalid email or password.")
        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("🔑 Demo credentials"):
            st.markdown("""
**Admin:** admin@rescue.com / admin123
**Operator:** operator@rescue.com / operator123
**User:** user@rescue.com / user123
**Drivers:** rajesh@rescue.com · amit@rescue.com · vijay@rescue.com ·
karan@rescue.com · nilesh@rescue.com — all use `driver123`
            """)


def logout():
    st.session_state.pop("user", None)
    safe_rerun()


# ============================================================================
# 6. SHARED UI PIECES
# ============================================================================

def hero(title, subtitle):
    st.markdown(f"""
    <div class="rr-hero">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def kpi_row(items):
    cols = st.columns(len(items))
    for c, (label, value) in zip(cols, items):
        with c:
            st.markdown(f"""
            <div class="rr-kpi">
                <div class="val">{value}</div>
                <div class="lbl">{label}</div>
            </div>
            """, unsafe_allow_html=True)


def marker_color(status):
    return {"Available": "green", "Busy": "red", "Maintenance": "orange", "Unavailable": "gray"}.get(status, "blue")


def location_picker(state_prefix, default_lat, default_lon, height=320):
    """
    Renders a click-to-pick OpenStreetMap map (same real-map style as the driver views)
    and returns the currently selected (lat, lon) for use in a nearby form.
    The pick is persisted in st.session_state under f"{state_prefix}_lat" / f"{state_prefix}_lon"
    so it survives reruns until the caller clears it (e.g. after a successful save).
    """
    lat_key, lon_key = f"{state_prefix}_lat", f"{state_prefix}_lon"
    if lat_key not in st.session_state:
        st.session_state[lat_key] = default_lat
    if lon_key not in st.session_state:
        st.session_state[lon_key] = default_lon

    st.caption("📍 Click anywhere on the map to set the exact location — or fine-tune the "
               "latitude/longitude fields below.")
    pick_map = folium.Map(
        location=[st.session_state[lat_key], st.session_state[lon_key]],
        zoom_start=13, tiles="OpenStreetMap",
    )
    folium.Marker(
        [st.session_state[lat_key], st.session_state[lon_key]],
        tooltip="📍 Selected Location", icon=folium.Icon(color="red", icon="warning-sign"),
    ).add_to(pick_map)
    result = st_folium(pick_map, width=None, height=height, key=f"{state_prefix}_picker_map")

    if result and result.get("last_clicked"):
        clicked_lat = result["last_clicked"]["lat"]
        clicked_lon = result["last_clicked"]["lng"]
        if round(clicked_lat, 6) != round(st.session_state[lat_key], 6) or \
           round(clicked_lon, 6) != round(st.session_state[lon_key], 6):
            st.session_state[lat_key] = clicked_lat
            st.session_state[lon_key] = clicked_lon
            safe_rerun()

    st.info(f"📍 Selected location: {st.session_state[lat_key]:.6f}, {st.session_state[lon_key]:.6f}")
    return st.session_state[lat_key], st.session_state[lon_key]


def clear_location_pick(state_prefix):
    st.session_state.pop(f"{state_prefix}_lat", None)
    st.session_state.pop(f"{state_prefix}_lon", None)


def build_fleet_map(df_amb, df_hosp=None, center=AHMEDABAD_CENTER, extra_points=None):
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB dark_matter")
    for _, r in df_amb.iterrows():
        if pd.isna(r["latitude"]) or pd.isna(r["longitude"]):
            continue
        popup = f"🚑 {r['ambulance_id']}<br>Driver: {r['driver']}<br>Status: {r['status']}<br>Type: {r['ambulance_type']}"
        folium.Marker(
            [r["latitude"], r["longitude"]],
            popup=popup,
            tooltip=r["ambulance_id"],
            icon=folium.Icon(color=marker_color(r["status"]), icon="plus-sign"),
        ).add_to(m)
    if df_hosp is not None:
        for _, r in df_hosp.iterrows():
            if pd.isna(r["latitude"]) or pd.isna(r["longitude"]):
                continue
            popup = f"🏥 {r['name']}<br>Beds: {r['beds']}"
            folium.Marker(
                [r["latitude"], r["longitude"]],
                popup=popup, tooltip=r["name"],
                icon=folium.Icon(color="blue", icon="plus", prefix="fa"),
            ).add_to(m)
    if extra_points:
        for pt in extra_points:
            folium.Marker(
                [pt["lat"], pt["lon"]], popup=pt.get("label", ""),
                icon=folium.Icon(color=pt.get("color", "purple"), icon=pt.get("icon", "info-sign")),
            ).add_to(m)
    return m


# ============================================================================
# 7. SIDEBAR / NAVIGATION
# ============================================================================

NAV = {
    "Admin": ["🏠 Dashboard", "🚑 Ambulance Fleet", "🏥 Hospitals", "🚨 Emergencies",
              "📊 Analytics", "⚙️ Management", "👥 Users"],
    "Operator": ["🏠 Dashboard", "🚨 Dispatch Center", "🚑 Ambulance Fleet", "🏥 Hospitals",
                 "📜 History", "📊 Analytics"],
    "Driver": ["🏠 Driver Console", "🗺️ Hospital Route", "📜 My History"],
    "User": ["🏠 Home", "🚨 Request Ambulance", "📍 Track Ambulance",
             "🏥 Select Hospital", "📜 My History"],
}


def sidebar():
    user = st.session_state["user"]
    with st.sidebar:
        st.markdown(f"### 🚑 RescueRoute")
        st.markdown(f"**{user['name']}**  \n{badge(user['role'], 'red')}", unsafe_allow_html=True)
        st.divider()
        choice = st.radio("Navigate", NAV[user["role"]], label_visibility="collapsed")
        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            logout()
    return choice


# ============================================================================
# 8. ADMIN PAGES
# ============================================================================

def admin_dashboard():
    hero("🏠 Admin Dashboard", "Live overview of the entire RescueRoute fleet & operations")
    amb = read("SELECT * FROM ambulances")
    hosp = read("SELECT * FROM hospitals")
    em = read("SELECT * FROM emergencies")

    kpi_row([
        ("Total Ambulances", len(amb)),
        ("Available", (amb["status"] == "Available").sum() if not amb.empty else 0),
        ("Busy", (amb["status"] == "Busy").sum() if not amb.empty else 0),
        ("Maintenance", (amb["status"] == "Maintenance").sum() if not amb.empty else 0),
    ])
    st.write("")
    kpi_row([
        ("Total Hospitals", len(hosp)),
        ("Active Emergencies", em["status"].isin(
            ["Requested", "Dispatched", "On the Way", "At Patient", "Hospital Selected"]
        ).sum() if not em.empty else 0),
        ("Completed", (em["status"] == "Completed").sum() if not em.empty else 0),
        ("Cancelled", (em["status"] == "Cancelled").sum() if not em.empty else 0),
    ])

    st.write("")
    st.markdown("#### 🗺️ Live Fleet Map")
    st_folium(build_fleet_map(amb, hosp), width=None, height=460, key="admin_dash_map")


def ambulance_fleet_page():
    hero("🚑 Ambulance Fleet", "Real-time fleet table and map — always reading from SQLite")
    amb = read("SELECT * FROM ambulances ORDER BY ambulance_id")

    if amb.empty:
        st.info("No ambulances yet. Add one from ⚙️ Management.")
        return

    tab1, tab2 = st.tabs(["📋 Fleet Table", "🗺️ Fleet Map"])
    with tab1:
        show = amb.copy()
        show["status"] = show["status"].apply(lambda s: badge(s, STATUS_BADGE.get(s, "gray")))
        st.write(
            show[["ambulance_id", "driver", "driver_email", "phone", "ambulance_type",
                  "status", "latitude", "longitude", "updated_at"]].to_html(escape=False, index=False),
            unsafe_allow_html=True,
        )
    with tab2:
        st_folium(build_fleet_map(amb), width=None, height=500, key="fleet_map_page")


def hospital_page(readonly=True):
    hero("🏥 Hospitals", "Hospital network — availability, ICU, and bed capacity")
    hosp = read("SELECT * FROM hospitals ORDER BY name")
    if hosp.empty:
        st.info("No hospitals yet.")
        return
    tab1, tab2 = st.tabs(["📋 Hospital Table", "🗺️ Hospital Map"])
    with tab1:
        show = hosp.copy()
        show["emergency_available"] = show["emergency_available"].map({1: "Yes", 0: "No"})
        show["icu_available"] = show["icu_available"].map({1: "Yes", 0: "No"})
        st.dataframe(
            show[["name", "address", "contact", "beds", "icu_available",
                  "emergency_available", "latitude", "longitude"]],
            use_container_width=True, hide_index=True,
        )
    with tab2:
        m = folium.Map(location=AHMEDABAD_CENTER, zoom_start=12, tiles="CartoDB dark_matter")
        for _, r in hosp.iterrows():
            folium.Marker(
                [r["latitude"], r["longitude"]],
                popup=f"🏥 {r['name']}<br>Beds: {r['beds']}<br>ICU: {'Yes' if r['icu_available'] else 'No'}",
                tooltip=r["name"],
                icon=folium.Icon(color="blue", icon="plus", prefix="fa"),
            ).add_to(m)
        st_folium(m, width=None, height=500, key="hospital_map_page")


def admin_emergencies_page():
    hero("🚨 Emergencies", "All emergency records across the platform")
    em = read("SELECT * FROM emergencies ORDER BY created_at DESC")
    render_emergency_table(em)


def render_emergency_table(em):
    if em.empty:
        st.info("No emergencies recorded yet.")
        return
    show = em.copy()
    show["severity_rank"] = show["severity"].map(SEVERITY_ORDER).fillna(9)
    show = show.sort_values(["severity_rank", "created_at"])
    show["status"] = show["status"].apply(lambda s: badge(s, STATUS_BADGE.get(s, "gray")))
    show["severity"] = show["severity"].apply(
        lambda s: badge(s, "red" if s == "Critical" else ("orange" if s == "High" else "blue"))
    )
    cols = ["id", "patient_name", "emergency_type", "severity", "ambulance_id",
            "hospital_id", "status", "traffic", "ambulance_distance", "hospital_distance",
            "eta", "created_at", "completed_at"]
    st.write(show[cols].to_html(escape=False, index=False), unsafe_allow_html=True)


def ambulance_management_page():
    hero("⚙️ Ambulance Management", "Add, edit, or remove ambulances from the fleet")
    show_flash()

    with st.expander("➕ Add New Ambulance", expanded=False):
        st.markdown("**📍 Set Ambulance Location on Map**")
        add_lat, add_lon = location_picker("add_amb", AHMEDABAD_CENTER[0], AHMEDABAD_CENTER[1])
        with st.form("add_ambulance_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                driver = st.text_input("Driver Name*")
                driver_email = st.text_input("Driver Login Email*")
                phone = st.text_input("Driver Phone*")
            with c2:
                atype = st.selectbox("Ambulance Type", AMBULANCE_TYPES)
                lat = st.number_input("Latitude", value=add_lat, format="%.6f",
                                       min_value=-90.0, max_value=90.0)
                lon = st.number_input("Longitude", value=add_lon, format="%.6f",
                                       min_value=-180.0, max_value=180.0)
            status = st.selectbox("Status", ["Available", "Maintenance"])
            submitted = st.form_submit_button("Add Ambulance", use_container_width=True)

            if submitted:
                if not driver or not driver_email or not phone:
                    st.error("Please fill all required (*) fields.")
                elif read_one("SELECT id FROM users WHERE email=?", (driver_email.strip().lower(),)):
                    st.error("This email is already registered to another user/driver.")
                else:
                    new_id = auto_ambulance_id()
                    try:
                        execute(
                            """INSERT INTO ambulances
                            (ambulance_id, driver, driver_email, phone, ambulance_type,
                             latitude, longitude, status, created_at, updated_at)
                            VALUES(?,?,?,?,?,?,?,?,?,?)""",
                            (new_id, driver, driver_email.strip().lower(), phone, atype,
                             lat, lon, status, now_str(), now_str()),
                        )
                        execute(
                            "INSERT INTO users(email,password,role,name,phone,created_at) VALUES(?,?,?,?,?,?)",
                            (driver_email.strip().lower(), "driver123", "Driver", driver, phone, now_str()),
                        )
                        clear_location_pick("add_amb")
                        flash(f"✅ {new_id} added and driver account created (password: driver123).")
                        safe_rerun()
                    except sqlite3.IntegrityError as e:
                        st.error(f"Could not add ambulance: {e}")

    st.markdown("#### ✏️ Edit / 🗑️ Delete Ambulance")
    amb = read("SELECT * FROM ambulances ORDER BY ambulance_id")
    if amb.empty:
        st.info("No ambulances to manage yet.")
        return

    selected = st.selectbox("Select ambulance", amb["ambulance_id"].tolist())
    row = amb[amb["ambulance_id"] == selected].iloc[0]

    st.markdown("**📍 Update Location on Map**")
    edit_lat, edit_lon = location_picker(f"edit_amb_{selected}", float(row["latitude"]), float(row["longitude"]))

    with st.form("edit_ambulance_form"):
        c1, c2 = st.columns(2)
        with c1:
            driver = st.text_input("Driver Name", value=row["driver"])
            driver_email = st.text_input("Driver Email", value=row["driver_email"])
            phone = st.text_input("Phone", value=row["phone"] or "")
        with c2:
            atype = st.selectbox("Ambulance Type", AMBULANCE_TYPES,
                                  index=AMBULANCE_TYPES.index(row["ambulance_type"])
                                  if row["ambulance_type"] in AMBULANCE_TYPES else 0)
            lat = st.number_input("Latitude", value=edit_lat, format="%.6f",
                                   min_value=-90.0, max_value=90.0)
            lon = st.number_input("Longitude", value=edit_lon, format="%.6f",
                                   min_value=-180.0, max_value=180.0)
        status = st.selectbox("Status", ["Available", "Busy", "Maintenance", "Unavailable"],
                               index=["Available", "Busy", "Maintenance", "Unavailable"].index(row["status"])
                               if row["status"] in ["Available", "Busy", "Maintenance", "Unavailable"] else 0)

        c3, c4 = st.columns(2)
        with c3:
            save = st.form_submit_button("💾 Save Changes", use_container_width=True)
        with c4:
            delete = st.form_submit_button("🗑️ Delete Ambulance", use_container_width=True)

        if save:
            execute(
                """UPDATE ambulances SET driver=?, driver_email=?, phone=?, ambulance_type=?,
                   latitude=?, longitude=?, status=?, updated_at=? WHERE ambulance_id=?""",
                (driver, driver_email.strip().lower(), phone, atype, lat, lon, status,
                 now_str(), selected),
            )
            clear_location_pick(f"edit_amb_{selected}")
            flash(f"✅ {selected} updated successfully.")
            safe_rerun()

        if delete:
            active = read_one(
                "SELECT id FROM emergencies WHERE ambulance_id=? AND status NOT IN ('Completed','Cancelled')",
                (selected,),
            )
            if active:
                st.error(f"❌ Cannot delete {selected} — it is linked to an active emergency.")
            else:
                execute("DELETE FROM ambulances WHERE ambulance_id=?", (selected,))
                check = read_one("SELECT COUNT(*) c FROM ambulances WHERE ambulance_id=?", (selected,))
                if check and check["c"] == 0:
                    clear_location_pick(f"edit_amb_{selected}")
                    flash(f"🗑️ {selected} deleted successfully.")
                else:
                    st.error("Deletion could not be verified.")
                safe_rerun()


def hospital_management_page():
    hero("🏥 Hospital Management", "Add, edit, or remove hospitals from the network")
    show_flash()

    with st.expander("➕ Add New Hospital"):
        st.markdown("**📍 Set Hospital Location on Map**")
        add_lat, add_lon = location_picker("add_hosp", AHMEDABAD_CENTER[0], AHMEDABAD_CENTER[1])
        with st.form("add_hospital_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input("Hospital Name*")
                address = st.text_input("Address")
                contact = st.text_input("Contact Number")
            with c2:
                lat = st.number_input("Latitude", value=add_lat, format="%.6f",
                                       min_value=-90.0, max_value=90.0)
                lon = st.number_input("Longitude", value=add_lon, format="%.6f",
                                       min_value=-180.0, max_value=180.0)
                beds = st.number_input("Beds", min_value=0, value=20, step=1)
            c3, c4 = st.columns(2)
            with c3:
                emergency_available = st.checkbox("Emergency Services Available", value=True)
            with c4:
                icu_available = st.checkbox("ICU Available", value=True)
            submitted = st.form_submit_button("Add Hospital", use_container_width=True)
            if submitted:
                if not name:
                    st.error("Hospital name is required.")
                else:
                    execute(
                        """INSERT INTO hospitals
                        (name, address, contact, latitude, longitude, emergency_available,
                         icu_available, beds, created_at, updated_at)
                        VALUES(?,?,?,?,?,?,?,?,?,?)""",
                        (name, address, contact, lat, lon, int(emergency_available),
                         int(icu_available), beds, now_str(), now_str()),
                    )
                    clear_location_pick("add_hosp")
                    flash(f"✅ Hospital '{name}' added.")
                    safe_rerun()

    st.markdown("#### ✏️ Edit / 🗑️ Delete Hospital")
    hosp = read("SELECT * FROM hospitals ORDER BY name")
    if hosp.empty:
        st.info("No hospitals to manage yet.")
        return

    selected = st.selectbox("Select hospital", hosp["name"].tolist())
    row = hosp[hosp["name"] == selected].iloc[0]

    st.markdown("**📍 Update Location on Map**")
    edit_lat, edit_lon = location_picker(f"edit_hosp_{int(row['id'])}", float(row["latitude"]), float(row["longitude"]))

    with st.form("edit_hospital_form"):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Hospital Name", value=row["name"])
            address = st.text_input("Address", value=row["address"] or "")
            contact = st.text_input("Contact", value=row["contact"] or "")
        with c2:
            lat = st.number_input("Latitude", value=edit_lat, format="%.6f",
                                   min_value=-90.0, max_value=90.0)
            lon = st.number_input("Longitude", value=edit_lon, format="%.6f",
                                   min_value=-180.0, max_value=180.0)
            beds = st.number_input("Beds", min_value=0, value=int(row["beds"]), step=1)
        c3, c4 = st.columns(2)
        with c3:
            emergency_available = st.checkbox("Emergency Available", value=bool(row["emergency_available"]))
        with c4:
            icu_available = st.checkbox("ICU Available", value=bool(row["icu_available"]))

        c5, c6 = st.columns(2)
        with c5:
            save = st.form_submit_button("💾 Save Changes", use_container_width=True)
        with c6:
            delete = st.form_submit_button("🗑️ Delete Hospital", use_container_width=True)

        if save:
            execute(
                """UPDATE hospitals SET name=?, address=?, contact=?, latitude=?, longitude=?,
                   emergency_available=?, icu_available=?, beds=?, updated_at=? WHERE id=?""",
                (name, address, contact, lat, lon, int(emergency_available), int(icu_available),
                 beds, now_str(), int(row["id"])),
            )
            clear_location_pick(f"edit_hosp_{int(row['id'])}")
            flash(f"✅ '{name}' updated successfully.")
            safe_rerun()

        if delete:
            active = read_one(
                "SELECT id FROM emergencies WHERE hospital_id=? AND status NOT IN ('Completed','Cancelled')",
                (int(row["id"]),),
            )
            if active:
                st.error(f"❌ Cannot delete '{selected}' — linked to an active emergency.")
            else:
                execute("DELETE FROM hospitals WHERE id=?", (int(row["id"]),))
                clear_location_pick(f"edit_hosp_{int(row['id'])}")
                flash(f"🗑️ '{selected}' deleted successfully.")
                safe_rerun()


def users_page():
    hero("👥 Users", "All registered accounts across roles")
    users = read("SELECT id, email, role, name, phone, created_at FROM users ORDER BY role, name")
    st.dataframe(users, use_container_width=True, hide_index=True)


def analytics_page():
    hero("📊 Analytics", "Operational insights across the platform")
    em = read("SELECT * FROM emergencies")
    amb = read("SELECT * FROM ambulances")

    total = len(em)
    critical = (em["severity"] == "Critical").sum() if total else 0
    completed = (em["status"] == "Completed").sum() if total else 0
    cancelled = (em["status"] == "Cancelled").sum() if total else 0
    avg_eta = round(em["eta"].dropna().mean(), 1) if total and em["eta"].notna().any() else 0
    avg_dist = round(em["ambulance_distance"].dropna().mean(), 2) if total and em["ambulance_distance"].notna().any() else 0
    busy_amb = (amb["status"] == "Busy").sum() if not amb.empty else 0
    utilization = round((busy_amb / len(amb)) * 100, 1) if not amb.empty else 0

    kpi_row([
        ("Total Emergencies", total), ("Critical", critical),
        ("Completed", completed), ("Cancelled", cancelled),
    ])
    st.write("")
    kpi_row([
        ("Avg ETA (min)", avg_eta), ("Avg Distance (km)", avg_dist),
        ("Fleet Utilization", f"{utilization}%"), ("Total Ambulances", len(amb)),
    ])

    if total == 0:
        st.info("No emergency data yet — charts will appear once emergencies are created.")
        return

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(em, names="emergency_type", title="Emergency Type Distribution", hole=0.45)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#eef2f7")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.bar(em["severity"].value_counts().reset_index(),
                       x="severity", y="count", title="Severity Distribution",
                       color="severity", color_discrete_map={"Critical": "#e63946", "High": "#f5a623", "Medium": "#4287f5"})
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#eef2f7")
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        traffic_counts = em["traffic"].dropna().value_counts().reset_index()
        if not traffic_counts.empty:
            fig3 = px.pie(traffic_counts, names="traffic", values="count", title="Traffic Distribution", hole=0.45)
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#eef2f7")
            st.plotly_chart(fig3, use_container_width=True)
    with c4:
        em2 = em.copy()
        em2["created_at"] = pd.to_datetime(em2["created_at"], errors="coerce")
        em2["date"] = em2["created_at"].dt.date
        trend = em2.groupby("date").size().reset_index(name="count")
        if not trend.empty:
            fig4 = px.line(trend, x="date", y="count", markers=True, title="Emergencies Over Time")
            fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#eef2f7")
            st.plotly_chart(fig4, use_container_width=True)


# ============================================================================
# 9. OPERATOR PAGES
# ============================================================================

def dispatch_center():
    hero("🚨 Dispatch Center", "Live queue of active emergencies — sorted by severity")
    show_flash()
    em = read("SELECT * FROM emergencies WHERE status NOT IN ('Completed','Cancelled')")
    if em.empty:
        st.success("No active emergencies right now. All clear ✅")
        return

    em["rank"] = em["severity"].map(SEVERITY_ORDER).fillna(9)
    em = em.sort_values(["rank", "created_at"])

    for _, r in em.iterrows():
        sev_kind = "red" if r["severity"] == "Critical" else ("orange" if r["severity"] == "High" else "blue")
        st.markdown(f"""
        <div class="rr-card">
            <b>Emergency #{r['id']}</b> — {badge(r['severity'], sev_kind)} {badge(r['status'], STATUS_BADGE.get(r['status'],'gray'))}<br>
            <b>Patient:</b> {r['patient_name']} &nbsp;|&nbsp; <b>Type:</b> {r['emergency_type']}<br>
            <b>Location:</b> {r['patient_lat']:.4f}, {r['patient_lon']:.4f}<br>
            <b>Ambulance:</b> {r['ambulance_id'] or '—'} &nbsp;|&nbsp; <b>ETA:</b> {r['eta'] or '—'} min
            &nbsp;|&nbsp; <b>Traffic:</b> {r['traffic'] or '—'}
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns([1, 1])
        with c1:
            if not r["ambulance_id"]:
                if st.button(f"🚑 Assign Nearest Available Ambulance", key=f"assign_{r['id']}"):
                    amb = nearest_available_ambulance(r["patient_lat"], r["patient_lon"])
                    if amb is None:
                        st.error("No ambulance currently available.")
                    else:
                        dist = haversine(r["patient_lat"], r["patient_lon"], amb["latitude"], amb["longitude"])
                        traffic = estimate_traffic()
                        eta = estimate_eta(dist, traffic)
                        execute("UPDATE ambulances SET status='Busy', updated_at=? WHERE ambulance_id=?",
                                (now_str(), amb["ambulance_id"]))
                        execute(
                            """UPDATE emergencies SET ambulance_id=?, status='Dispatched',
                               ambulance_distance=?, traffic=?, eta=?, updated_at=? WHERE id=?""",
                            (amb["ambulance_id"], round(dist, 2), traffic, eta, now_str(), int(r["id"])),
                        )
                        execute(
                            "INSERT INTO location_updates(emergency_id, ambulance_id, latitude, longitude, updated_at) VALUES(?,?,?,?,?)",
                            (int(r["id"]), amb["ambulance_id"], amb["latitude"], amb["longitude"], now_str()),
                        )
                        flash(f"✅ {amb['ambulance_id']} assigned to Emergency #{r['id']}.")
                        safe_rerun()
        with c2:
            options = [s for s in STATUS_FLOW if s != r["status"]]
            new_status = st.selectbox("Update status", options, key=f"status_{r['id']}")
            if st.button("Apply", key=f"apply_{r['id']}"):
                update_emergency_status(int(r["id"]), new_status, r["ambulance_id"])
                flash(f"Emergency #{r['id']} status updated to {new_status}.")
                safe_rerun()


def update_emergency_status(emergency_id, new_status, ambulance_id):
    completed_at = now_str() if new_status in ("Completed", "Cancelled") else None
    execute(
        "UPDATE emergencies SET status=?, updated_at=?, completed_at=COALESCE(?, completed_at) WHERE id=?",
        (new_status, now_str(), completed_at, emergency_id),
    )
    if new_status in ("Completed", "Cancelled") and ambulance_id:
        execute("UPDATE ambulances SET status='Available', updated_at=? WHERE ambulance_id=?",
                (now_str(), ambulance_id))


def history_page(role, user_email=None, ambulance_id=None):
    hero("📜 Emergency History", "Full record of emergencies")
    if role == "User":
        em = read("SELECT * FROM emergencies WHERE user_email=? ORDER BY created_at DESC", (user_email,))
    elif role == "Driver":
        em = read("SELECT * FROM emergencies WHERE ambulance_id=? ORDER BY created_at DESC", (ambulance_id,))
    else:
        em = read("SELECT * FROM emergencies ORDER BY created_at DESC")
    render_emergency_table(em)


# ============================================================================
# 10. DRIVER PAGES
# ============================================================================

def get_driver_ambulance(email):
    return read_one("SELECT * FROM ambulances WHERE driver_email=?", (email,))


def driver_console():
    user = st.session_state["user"]
    amb = get_driver_ambulance(user["email"])
    hero("🏠 Driver Console", f"Live status for {user['name']}")
    show_flash()

    if not amb:
        st.error("No ambulance is linked to your account. Please contact Admin.")
        return

    em = read_one(
        "SELECT * FROM emergencies WHERE ambulance_id=? AND status NOT IN ('Completed','Cancelled') ORDER BY created_at DESC",
        (amb["ambulance_id"],),
    )

    kpi_row([
        ("Ambulance", amb["ambulance_id"]),
        ("Type", amb["ambulance_type"]),
        ("Status", amb["status"]),
        ("Active Emergency", f"#{em['id']}" if em else "None"),
    ])

    if em:
        st.markdown(f"""
        <div class="rr-card">
            <b>Patient:</b> {em['patient_name']} ({em['patient_phone']})<br>
            <b>Emergency Type:</b> {em['emergency_type']} &nbsp;|&nbsp;
            <b>Severity:</b> {badge(em['severity'], 'red' if em['severity']=='Critical' else 'orange')}<br>
            <b>Location:</b> {em['patient_lat']:.4f}, {em['patient_lon']:.4f}<br>
            <b>ETA:</b> {em['eta']} min &nbsp;|&nbsp; <b>Status:</b> {badge(em['status'], STATUS_BADGE.get(em['status'],'gray'))}
        </div>
        """, unsafe_allow_html=True)

        real_route = get_real_route(amb["latitude"], amb["longitude"], em["patient_lat"], em["patient_lon"])

        m = folium.Map(location=[em["patient_lat"], em["patient_lon"]], zoom_start=13, tiles="OpenStreetMap")
        folium.Marker([amb["latitude"], amb["longitude"]], tooltip="🚑 Ambulance",
                      icon=folium.Icon(color="green", icon="plus-sign")).add_to(m)
        folium.Marker([em["patient_lat"], em["patient_lon"]], tooltip="🚨 Patient",
                      icon=folium.Icon(color="red", icon="warning-sign")).add_to(m)

        if real_route:
            folium.PolyLine(
                real_route["coordinates"], color="#e63946", weight=4, opacity=0.85,
                tooltip=f"🛣️ {real_route['distance_km']} km · {real_route['duration_min']} min",
            ).add_to(m)
            m.fit_bounds(real_route["coordinates"])
            st.caption(f"🛣️ Real road distance: {real_route['distance_km']} km · "
                       f"Estimated duration: {real_route['duration_min']} min (via OSRM)")
        else:
            m.fit_bounds([[amb["latitude"], amb["longitude"]], [em["patient_lat"], em["patient_lon"]]])
            st.warning("⚠️ Real road routing service is temporarily unavailable.")

        st_folium(m, width=None, height=420, key="driver_console_map")

        st.markdown("#### Update Status")
        c1, c2, c3, c4, c5 = st.columns(5)
        buttons = [
            ("🚨 Dispatched", "Dispatched", c1),
            ("🚗 On the Way", "On the Way", c2),
            ("📍 At Patient", "At Patient", c3),
            ("🏥 Hospital Selected", "Hospital Selected", c4),
            ("✅ Completed", "Completed", c5),
        ]
        for label, status_val, col in buttons:
            with col:
                if st.button(label, key=f"drv_{status_val}", use_container_width=True):
                    if status_val == "At Patient":
                        execute("UPDATE ambulances SET latitude=?, longitude=?, updated_at=? WHERE ambulance_id=?",
                                (em["patient_lat"], em["patient_lon"], now_str(), amb["ambulance_id"]))
                        execute(
                            "INSERT INTO location_updates(emergency_id, ambulance_id, latitude, longitude, updated_at) VALUES(?,?,?,?,?)",
                            (int(em["id"]), amb["ambulance_id"], em["patient_lat"], em["patient_lon"], now_str()),
                        )
                    update_emergency_status(int(em["id"]), status_val, amb["ambulance_id"])
                    flash(f"Status updated to {status_val}.")
                    safe_rerun()
    else:
        st.info("No active emergency assigned right now. Standing by.")


def driver_hospital_route_page():
    user = st.session_state["user"]
    amb = get_driver_ambulance(user["email"])
    hero("🗺️ Hospital Route", "Patient → Hospital route once a hospital is selected")

    if not amb:
        st.error("No ambulance linked to your account.")
        return

    em = read_one(
        "SELECT * FROM emergencies WHERE ambulance_id=? AND status IN ('Hospital Selected') ORDER BY created_at DESC",
        (amb["ambulance_id"],),
    )
    if not em or not em["hospital_id"]:
        st.info("No hospital has been selected for your current emergency yet.")
        return

    hosp = read_one("SELECT * FROM hospitals WHERE id=?", (int(em["hospital_id"]),))
    if not hosp:
        st.warning("Selected hospital could not be found (may have been removed).")
        return

    opts = route_options(em["patient_lat"], em["patient_lon"], hosp["latitude"], hosp["longitude"])
    st.markdown(f"#### 🚨 Patient → 🏥 {hosp['name']}")

    if not opts:
        st.warning("⚠️ Real road routing service is temporarily unavailable.")

    cols = st.columns(len(opts)) if opts else None
    for c, o in zip(cols or [], opts):
        with c:
            tag = "⭐ Recommended" if o["recommended"] else ""
            st.markdown(f"""
            <div class="rr-card" style="text-align:center; {'border-color:#4ee08a;' if o['recommended'] else ''}">
                <b>{o['route']}</b><br>{tag}<br>
                Road Distance: {o['distance']} km<br>
                Estimated Duration: {o['duration']} min
            </div>
            """, unsafe_allow_html=True)

    m = folium.Map(location=[em["patient_lat"], em["patient_lon"]], zoom_start=13, tiles="OpenStreetMap")
    folium.Marker([em["patient_lat"], em["patient_lon"]], tooltip="🚨 Patient",
                  icon=folium.Icon(color="red", icon="warning-sign")).add_to(m)
    folium.Marker([hosp["latitude"], hosp["longitude"]], tooltip=f"🏥 {hosp['name']}",
                  icon=folium.Icon(color="blue", icon="plus", prefix="fa")).add_to(m)

    if opts:
        best = opts[0]
        folium.PolyLine(
            best["coordinates"], color="#e63946", weight=5, opacity=0.9,
            tooltip=f"🛣️ {best['distance']} km · {best['duration']} min",
        ).add_to(m)
        m.fit_bounds(best["coordinates"])
    else:
        m.fit_bounds([[em["patient_lat"], em["patient_lon"]], [hosp["latitude"], hosp["longitude"]]])

    st_folium(m, width=None, height=440, key="driver_route_map")

    if st.button("✅ Mark Emergency Completed", use_container_width=True):
        update_emergency_status(int(em["id"]), "Completed", amb["ambulance_id"])
        flash("Emergency marked completed. Ambulance is now Available.")
        safe_rerun()


# ============================================================================
# 11. USER PAGES
# ============================================================================

def user_home():
    user = st.session_state["user"]
    hero("🏠 My Emergencies", f"Welcome, {user['name']}")
    show_flash()

    em = read(
        "SELECT * FROM emergencies WHERE user_email=? ORDER BY created_at DESC LIMIT 1",
        (user["email"],),
    )
    if em.empty:
        st.info("You have no emergency requests yet. Use 🚨 Request Ambulance to create one.")
        return

    r = em.iloc[0]
    amb = read_one("SELECT * FROM ambulances WHERE ambulance_id=?", (r["ambulance_id"],)) if r["ambulance_id"] else None
    hosp = read_one("SELECT * FROM hospitals WHERE id=?", (int(r["hospital_id"]),)) if r["hospital_id"] else None

    st.markdown(f"""
    <div class="rr-card">
        <h4>Emergency #{r['id']} — {badge(r['status'], STATUS_BADGE.get(r['status'],'gray'))}</h4>
        <b>Ambulance:</b> {amb['ambulance_id'] if amb else 'Not yet assigned'} &nbsp;|&nbsp;
        <b>Driver:</b> {amb['driver'] if amb else '—'}<br>
        <b>ETA:</b> {r['eta'] or '—'} min &nbsp;|&nbsp; <b>Ambulance Distance:</b> {r['ambulance_distance'] or '—'} km<br>
        <b>Hospital:</b> {hosp['name'] if hosp else 'Not yet selected'}
    </div>
    """, unsafe_allow_html=True)


def request_ambulance_page():
    user = st.session_state["user"]
    hero("🚨 Request Ambulance", "Fill in the emergency details below")

    active = read_one(
        "SELECT id FROM emergencies WHERE user_email=? AND status NOT IN ('Completed','Cancelled')",
        (user["email"],),
    )
    if active:
        st.warning(f"You already have an active emergency (#{active['id']}). "
                   "Please wait for it to complete, or cancel it via history, before creating a new one.")
        return

    if "req_lat" not in st.session_state:
        st.session_state["req_lat"] = AHMEDABAD_CENTER[0]
    if "req_lon" not in st.session_state:
        st.session_state["req_lon"] = AHMEDABAD_CENTER[1]

    st.markdown("#### 📍 Select Patient Location on Map")
    st.caption("Click anywhere on the real road map to pin the exact location — this avoids typing "
               "mistakes in the latitude/longitude boxes below. You can still fine-tune them manually.")

    pick_map = folium.Map(
        location=[st.session_state["req_lat"], st.session_state["req_lon"]],
        zoom_start=13, tiles="OpenStreetMap",
    )
    folium.Marker(
        [st.session_state["req_lat"], st.session_state["req_lon"]],
        tooltip="📍 Selected Location", icon=folium.Icon(color="red", icon="warning-sign"),
    ).add_to(pick_map)
    map_result = st_folium(pick_map, width=None, height=380, key="request_location_map")

    if map_result and map_result.get("last_clicked"):
        clicked_lat = map_result["last_clicked"]["lat"]
        clicked_lon = map_result["last_clicked"]["lng"]
        if round(clicked_lat, 6) != round(st.session_state["req_lat"], 6) or \
           round(clicked_lon, 6) != round(st.session_state["req_lon"], 6):
            st.session_state["req_lat"] = clicked_lat
            st.session_state["req_lon"] = clicked_lon
            safe_rerun()

    st.info(f"📍 Selected location: {st.session_state['req_lat']:.6f}, {st.session_state['req_lon']:.6f}")

    with st.form("request_form"):
        c1, c2 = st.columns(2)
        with c1:
            patient_name = st.text_input("Patient Name*", value=user.get("name", ""))
            patient_phone = st.text_input("Phone*", value=user.get("phone", "") or "")
            emergency_type = st.selectbox("Emergency Type", EMERGENCY_TYPES)
        with c2:
            severity = st.selectbox("Severity", SEVERITY_LEVELS)
            lat = st.number_input(
                "Patient Latitude*", value=st.session_state["req_lat"], format="%.6f",
                min_value=-90.0, max_value=90.0,
                help="Auto-filled from the map above. Avoid clearing this box completely — an empty "
                     "box resets to 0, which is an invalid location. Click the map instead to change it.",
            )
            lon = st.number_input(
                "Patient Longitude*", value=st.session_state["req_lon"], format="%.6f",
                min_value=-180.0, max_value=180.0,
                help="Auto-filled from the map above. Avoid clearing this box completely — an empty "
                     "box resets to 0, which is an invalid location. Click the map instead to change it.",
            )

        submitted = st.form_submit_button("🚨 Submit Emergency Request", use_container_width=True)

        if submitted:
            if not patient_name or not patient_phone:
                st.error("Please fill all required (*) fields.")
            elif not (-90 <= lat <= 90 and -180 <= lon <= 180):
                st.error("Invalid coordinates provided.")
            elif lat == 0.0 and lon == 0.0:
                st.error("⚠️ Latitude/Longitude looks empty (auto-filled to 0, 0). "
                         "Please click a location on the map above, or re-enter valid coordinates.")
            else:
                amb = nearest_available_ambulance(lat, lon)
                if amb is None:
                    execute(
                        """INSERT INTO emergencies
                        (user_email, patient_name, patient_phone, emergency_type, severity,
                         patient_lat, patient_lon, status, created_at, updated_at)
                        VALUES(?,?,?,?,?,?,?,?,?,?)""",
                        (user["email"], patient_name, patient_phone, emergency_type, severity,
                         lat, lon, "Requested", now_str(), now_str()),
                    )
                    st.warning("🚫 No ambulance currently available. Emergency added to dispatch queue.")
                else:
                    dist = haversine(lat, lon, amb["latitude"], amb["longitude"])
                    traffic = estimate_traffic()
                    eta = estimate_eta(dist, traffic)
                    eid = execute(
                        """INSERT INTO emergencies
                        (user_email, patient_name, patient_phone, emergency_type, severity,
                         patient_lat, patient_lon, ambulance_id, status, traffic,
                         ambulance_distance, eta, created_at, updated_at)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (user["email"], patient_name, patient_phone, emergency_type, severity,
                         lat, lon, amb["ambulance_id"], "Dispatched", traffic,
                         round(dist, 2), eta, now_str(), now_str()),
                    )
                    execute("UPDATE ambulances SET status='Busy', updated_at=? WHERE ambulance_id=?",
                            (now_str(), amb["ambulance_id"]))
                    execute(
                        "INSERT INTO location_updates(emergency_id, ambulance_id, latitude, longitude, updated_at) VALUES(?,?,?,?,?)",
                        (eid, amb["ambulance_id"], amb["latitude"], amb["longitude"], now_str()),
                    )
                    st.success(f"✅ Ambulance {amb['ambulance_id']} assigned! ETA: {eta} min.")
                flash("Emergency request submitted.")
                safe_rerun()


def track_ambulance_page():
    user = st.session_state["user"]
    hero("📍 Track Ambulance", "Live location of your assigned ambulance")

    em = read_one(
        "SELECT * FROM emergencies WHERE user_email=? AND status NOT IN ('Completed','Cancelled') ORDER BY created_at DESC",
        (user["email"],),
    )
    if not em or not em["ambulance_id"]:
        st.info("No ambulance currently assigned to you.")
        return

    amb = read_one("SELECT * FROM ambulances WHERE ambulance_id=?", (em["ambulance_id"],))
    if not amb:
        st.warning("Assigned ambulance record not found.")
        return

    kpi_row([
        ("Ambulance", amb["ambulance_id"]), ("Driver", amb["driver"]),
        ("Status", amb["status"]), ("ETA (min)", em["eta"] or "—"),
    ])

    if st.button("🔄 Refresh Tracking"):
        safe_rerun()

    m = folium.Map(location=[em["patient_lat"], em["patient_lon"]], zoom_start=13, tiles="OpenStreetMap")
    folium.Marker([amb["latitude"], amb["longitude"]], tooltip="🚑 Ambulance",
                  icon=folium.Icon(color="green", icon="plus-sign")).add_to(m)
    folium.Marker([em["patient_lat"], em["patient_lon"]], tooltip="🚨 You",
                  icon=folium.Icon(color="red", icon="warning-sign")).add_to(m)

    real_route = get_real_route(amb["latitude"], amb["longitude"], em["patient_lat"], em["patient_lon"])
    if real_route:
        folium.PolyLine(
            real_route["coordinates"], color="#e63946", weight=4, opacity=0.85,
            tooltip=f"🛣️ {real_route['distance_km']} km · {real_route['duration_min']} min",
        ).add_to(m)
        m.fit_bounds(real_route["coordinates"])
        st.caption(f"🛣️ Real road distance: {real_route['distance_km']} km · "
                   f"Estimated duration: {real_route['duration_min']} min (via OSRM)")
    else:
        m.fit_bounds([[amb["latitude"], amb["longitude"]], [em["patient_lat"], em["patient_lon"]]])
        st.warning("⚠️ Real road routing service is temporarily unavailable.")

    st_folium(m, width=None, height=440, key="user_track_map")


def select_hospital_page():
    user = st.session_state["user"]
    hero("🏥 Select Hospital", "Choose a hospital within 5 KM of the patient")

    em = read_one(
        "SELECT * FROM emergencies WHERE user_email=? AND status='At Patient' ORDER BY created_at DESC",
        (user["email"],),
    )
    if not em:
        st.info("Hospital selection becomes available once the ambulance has reached the patient "
                "(status: At Patient).")
        return

    hosp = nearby_hospitals(em["patient_lat"], em["patient_lon"], radius_km=5.0)
    if hosp.empty:
        st.warning("🚫 No hospitals found within 5 KM of your location.")
        return

    m = folium.Map(location=[em["patient_lat"], em["patient_lon"]], zoom_start=13, tiles="OpenStreetMap")
    folium.Marker([em["patient_lat"], em["patient_lon"]], tooltip="🚨 Patient",
                  icon=folium.Icon(color="red", icon="warning-sign")).add_to(m)
    folium.Circle([em["patient_lat"], em["patient_lon"]], radius=5000, color="#e63946",
                  fill=True, fill_opacity=0.05).add_to(m)
    for _, r in hosp.iterrows():
        folium.Marker([r["latitude"], r["longitude"]], tooltip=r["name"],
                      icon=folium.Icon(color="blue", icon="plus", prefix="fa")).add_to(m)
    st_folium(m, width=None, height=420, key="select_hospital_map")

    for _, r in hosp.iterrows():
        st.markdown(f"""
        <div class="rr-card">
            <b>🏥 {r['name']}</b> — Distance: {r['dist']:.2f} KM<br>
            {r['address'] or ''}<br>
            ICU: {'Available' if r['icu_available'] else 'Not Available'} &nbsp;|&nbsp; Beds: {r['beds']}
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"Select {r['name']}", key=f"sel_hosp_{r['id']}"):
            traffic = estimate_traffic()
            eta = estimate_eta(r["dist"], traffic)
            execute(
                """UPDATE emergencies SET hospital_id=?, hospital_distance=?, traffic=?, eta=?,
                   status='Hospital Selected', updated_at=? WHERE id=?""",
                (int(r["id"]), round(r["dist"], 2), traffic, eta, now_str(), int(em["id"])),
            )
            flash(f"✅ {r['name']} selected as destination hospital.")
            safe_rerun()


# ============================================================================
# 12. MAIN
# ============================================================================

def main():
    init_db()
    inject_css()

    if "user" not in st.session_state:
        login_page()
        return

    user = st.session_state["user"]
    choice = sidebar()
    show_flash()

    role = user["role"]

    try:
        if role == "Admin":
            if choice == "🏠 Dashboard":
                admin_dashboard()
            elif choice == "🚑 Ambulance Fleet":
                ambulance_fleet_page()
            elif choice == "🏥 Hospitals":
                hospital_page()
            elif choice == "🚨 Emergencies":
                admin_emergencies_page()
            elif choice == "📊 Analytics":
                analytics_page()
            elif choice == "⚙️ Management":
                tab1, tab2 = st.tabs(["🚑 Ambulance Management", "🏥 Hospital Management"])
                with tab1:
                    ambulance_management_page()
                with tab2:
                    hospital_management_page()
            elif choice == "👥 Users":
                users_page()

        elif role == "Operator":
            if choice == "🏠 Dashboard":
                admin_dashboard()
            elif choice == "🚨 Dispatch Center":
                dispatch_center()
            elif choice == "🚑 Ambulance Fleet":
                ambulance_fleet_page()
            elif choice == "🏥 Hospitals":
                hospital_page()
            elif choice == "📜 History":
                history_page("Operator")
            elif choice == "📊 Analytics":
                analytics_page()

        elif role == "Driver":
            if choice == "🏠 Driver Console":
                driver_console()
            elif choice == "🗺️ Hospital Route":
                driver_hospital_route_page()
            elif choice == "📜 My History":
                amb = get_driver_ambulance(user["email"])
                history_page("Driver", ambulance_id=amb["ambulance_id"] if amb else None)

        elif role == "User":
            if choice == "🏠 Home":
                user_home()
            elif choice == "🚨 Request Ambulance":
                request_ambulance_page()
            elif choice == "📍 Track Ambulance":
                track_ambulance_page()
            elif choice == "🏥 Select Hospital":
                select_hospital_page()
            elif choice == "📜 My History":
                history_page("User", user_email=user["email"])
        else:
            st.error("Unknown role. Please contact Admin.")

    except sqlite3.Error as e:
        st.error(f"A database error occurred: {e}")
    except Exception as e:
        st.error("Something went wrong while loading this page. Please try again.")
        with st.expander("Technical details (for developer)"):
            st.code(str(e))


if __name__ == "__main__":
    main()