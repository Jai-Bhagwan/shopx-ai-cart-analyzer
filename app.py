import streamlit as st
import pandas as pd
import numpy as np
import re
import shap
import requests
import urllib.parse
from io import StringIO
from sklearn.ensemble import RandomForestClassifier

# 1. Page Setup
st.set_page_config(
    page_title="ShopX AI — Cart Abandonment & Recovery Engine",
    page_icon="🛒",
    layout="wide"
)

# Custom Dribbble-Style Bento Glassmorphism Theme (Charcoal + Lime/Teal Ambient Glow)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* Ambient Charcoal Background with Lime-Olive & Warm Glows */
    .stApp {
        background-color: #111312;
        background-image: 
            radial-gradient(circle at 12% 15%, rgba(163, 217, 58, 0.14) 0%, transparent 38%),
            radial-gradient(circle at 88% 82%, rgba(245, 158, 11, 0.11) 0%, transparent 40%),
            radial-gradient(circle at 55% 45%, rgba(45, 212, 191, 0.07) 0%, transparent 45%);
        color: #f3f4f6;
    }

    /* Top Floating Glass Pill Header */
    .bento-navbar {
        background: rgba(28, 31, 30, 0.72);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 50px;
        padding: 12px 26px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    .nav-brand {
        font-size: 1.25rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.3px;
    }
    .nav-pills {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }
    .nav-pill-active {
        background: #d4f34a;
        color: #111312;
        font-weight: 700;
        font-size: 0.8rem;
        padding: 6px 16px;
        border-radius: 30px;
    }
    .nav-pill {
        background: rgba(255, 255, 255, 0.06);
        color: #cbd5e1;
        font-weight: 500;
        font-size: 0.8rem;
        padding: 6px 14px;
        border-radius: 30px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Bento Glass KPI Cards (Inspired by Progress Statistics Cards) */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(34, 38, 36, 0.78), rgba(22, 25, 24, 0.68));
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 20px 22px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.38);
        transition: all 0.25s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: rgba(212, 243, 74, 0.45);
        box-shadow: 0 16px 38px rgba(212, 243, 74, 0.08);
    }
    [data-testid="stMetricLabel"] {
        color: #9ca3af !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
    }

    /* Color-Coded Accents per KPI Card */
    div[data-testid="column"]:nth-of-type(1) [data-testid="stMetricValue"] { color: #d4f34a !important; }
    div[data-testid="column"]:nth-of-type(2) [data-testid="stMetricValue"] { color: #f5b041 !important; }
    div[data-testid="column"]:nth-of-type(3) [data-testid="stMetricValue"] { color: #4ce0b3 !important; }
    div[data-testid="column"]:nth-of-type(4) [data-testid="stMetricValue"] { color: #a78bfa !important; }

    /* Bento Glass Containers for Charts & Panels */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(28, 31, 30, 0.65) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 24px !important;
        padding: 18px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35) !important;
    }

    /* Subheaders */
    h2, h3 {
        color: #f3f4f6 !important;
        font-weight: 700 !important;
        letter-spacing: -0.3px;
    }

    /* Lime-Pill Action Button */
    .stButton > button {
        background: #d4f34a !important;
        color: #111312 !important;
        border: none !important;
        border-radius: 30px !important;
        padding: 0.45rem 1.3rem !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        box-shadow: 0 4px 16px rgba(212, 243, 74, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: #e2fa70 !important;
        transform: scale(1.03);
    }

    /* Recovery Card & Inputs */
    [data-testid="stAlert"] {
        background: rgba(34, 38, 36, 0.78) !important;
        border: 1px solid rgba(76, 224, 179, 0.3) !important;
        border-radius: 20px !important;
        color: #e2e8f0 !important;
    }
    textarea, input {
        background-color: rgba(20, 22, 21, 0.85) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 16px !important;
    }

    /* Pill Action Links for Gmail & WhatsApp */
    .action-btn-gmail {
        display: inline-block;
        background: #d4f34a;
        color: #111312 !important;
        font-weight: 700;
        padding: 9px 18px;
        border-radius: 30px;
        text-decoration: none !important;
        margin-right: 10px;
        font-size: 0.88rem;
    }
    .action-btn-wa {
        display: inline-block;
        background: #4ce0b3;
        color: #111312 !important;
        font-weight: 700;
        padding: 9px 18px;
        border-radius: 30px;
        text-decoration: none !important;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)

# 2. Config: Tumhari Live Google Sheet ID
SHEET_ID = "1oZZ3L780Bx21aUP-nrb3d33_C9cQwsMd_mqn6AgK26g"
LOCAL_EXCEL_FILE = "Shopx Store Data.xlsx"


# 3. Data Loader (5-sec timeout + automatic local Excel fallback)
@st.cache_data(ttl=60)
def load_live_data(1oZZ3L780Bx21aUP-nrb3d33_C9cQwsMd_mqn6AgK26g):
    df = None
    source_used = "Local Excel Backup"

    if sheet_id and sheet_id != "1oZZ3L780Bx21aUP-nrb3d33_C9cQwsMd_mqn6AgK26g":
        try:
            url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text), header=None)
            source_used = "Live Google Sheet"
        except Exception as e:
            st.warning(f"Live Google Sheet fetch nahi ho payi ({e}). Local Excel load kar rahe hain...")

    if df is None:
        df = pd.read_excel(LOCAL_EXCEL_FILE, header=None)

    df = df.iloc[:, :8].copy()
    df.columns = [
        'Timestamp', 'Event_Type', 'User_Name', 'Email',
        'Details', 'Price_Raw', 'IP_Address', 'Device_Browser'
    ]
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp']).reset_index(drop=True)
    return df, source_used


# 4. Price Parser
def parse_price(val):
    if pd.isna(val) or str(val).strip() in ['—', '-', '']:
        return np.nan
    cleaned = re.sub(r'[₹,\s]', '', str(val))
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


# 5. ML + SHAP Root-Cause Pipeline
def run_ai_pipeline(df):
    df['Price'] = df['Price_Raw'].apply(parse_price)
    cart_df = df[df['Event_Type'] == 'ADD_TO_CART'].copy()
    order_df = df[df['Event_Type'] == 'ORDER_PLACED'].copy()

    if cart_df.empty:
        return pd.DataFrame()

    user_email_map = (
        df[df['Event_Type'].isin(['SIGNUP', 'LOGIN', 'ORDER_PLACED'])]
        .dropna(subset=['Email'])
        .groupby('User_Name')['Email']
        .first()
        .to_dict()
    )

    cart_df['Product'] = (
        cart_df['Details']
        .astype(str)
        .str.replace(r'^ADDED:\s*', '', regex=True)
        .str.replace(r'\s*\(x\d+\)$', '', regex=True)
        .str.strip()
    )
    cart_df = cart_df[cart_df['Product'].str.lower() != 'nan'].copy()

    def check_purchased(row):
        user_orders = order_df[order_df['User_Name'] == row['User_Name']]
        prefix = str(row['Product'])[:25].lower()
        for _, o in user_orders.iterrows():
            if prefix in str(o['Details']).lower() or (pd.notna(o['Price']) and o['Price'] == row['Price']):
                return 1
        return 0

    cart_df['Purchased'] = cart_df.apply(check_purchased, axis=1)
    cart_df['Contact_Info'] = (
        cart_df['Email']
        .fillna(cart_df['User_Name'].map(user_email_map))
        .fillna("Not Captured")
    )

    def extract_contact_details(contact_str):
        text = str(contact_str)
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        email_val = email_match.group(0) if email_match else ""

        phone_match = re.search(r'(?:\+91[\-\s]?|91[\-\s]?|0)?([6-9]\d{9})\b', text)
        phone_val = f"91{phone_match.group(1)}" if phone_match else ""
        return email_val, phone_
