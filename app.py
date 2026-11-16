import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
import urllib.parse
from io import StringIO
from sklearn.ensemble import RandomForestClassifier

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# 1. Page Setup
st.set_page_config(
    page_title="ShopX AI — Cart Abandonment & Recovery Engine",
    page_icon="🛒",
    layout="wide"
)

# Custom Dribbble-Style Bento Glassmorphism Theme
st.markdown("""
<style>
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Ambient Charcoal Background with Lime-Olive & Warm Amber Glows */
    .stApp {
        background-color: #111312 !important;
        background-image: 
            radial-gradient(circle at 12% 15%, rgba(163, 217, 58, 0.14) 0%, transparent 38%),
            radial-gradient(circle at 88% 82%, rgba(245, 158, 11, 0.12) 0%, transparent 40%),
            radial-gradient(circle at 55% 45%, rgba(45, 212, 191, 0.08) 0%, transparent 45%) !important;
        color: #f3f4f6 !important;
    }

    /* Make all widget labels clearly visible */
    label, .stSelectbox label, .stTextArea label, .stTextInput label, [data-testid="stWidgetLabel"] p {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        margin-bottom: 6px !important;
    }

    /* Top Floating Glass Pill Header */
    .bento-navbar {
        background: rgba(28, 31, 30, 0.75);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 50px;
        padding: 12px 26px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
    }
    .nav-brand {
        font-size: 1.25rem;
        font-weight: 800;
        color: #ffffff;
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
        background: rgba(255, 255, 255, 0.07);
        color: #cbd5e1;
        font-weight: 500;
        font-size: 0.8rem;
        padding: 6px 14px;
        border-radius: 30px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* Bento Glass KPI Cards */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(34, 38, 36, 0.75), rgba(22, 25, 24, 0.65));
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 22px;
        padding: 16px 18px;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.35);
    }
    [data-testid="stMetricLabel"] p { color: #9ca3af !important; font-weight: 600 !important; font-size: 0.82rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.75rem !important; font-weight: 800 !important; }
    div[data-testid="column"]:nth-of-type(1) [data-testid="stMetricValue"] { color: #d4f34a !important; }
    div[data-testid="column"]:nth-of-type(2) [data-testid="stMetricValue"] { color: #f5b041 !important; }
    div[data-testid="column"]:nth-of-type(3) [data-testid="stMetricValue"] { color: #4ce0b3 !important; }
    div[data-testid="column"]:nth-of-type(4) [data-testid="stMetricValue"] { color: #a78bfa !important; }

    /* Bento Glass Containers */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(28, 31, 30, 0.62) !important;
        backdrop-filter: blur(18px) !important;
        -webkit-backdrop-filter: blur(18px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 24px !important;
        padding: 18px !important;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35) !important;
    }

    /* Frosted Glass Customer Details Slab */
    .customer-glass-slab {
        background: linear-gradient(135deg, rgba(42, 46, 44, 0.68), rgba(24, 27, 26, 0.72));
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 18px 20px;
        margin: 12px 0 16px 0;
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.08), 0 8px 24px rgba(0, 0, 0, 0.3);
        color: #f1f5f9;
        line-height: 1.65;
        font-size: 0.92rem;
    }
    .slab-badge-lime {
        display: inline-block;
        background: rgba(212, 243, 74, 0.16);
        color: #d4f34a;
        border: 1px solid rgba(212, 243, 74, 0.35);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .slab-badge-teal {
        display: inline-block;
        background: rgba(76, 224, 179, 0.16);
        color: #4ce0b3;
        border: 1px solid rgba(76, 224, 179, 0.35);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-left: 6px;
    }

    /* Dark Glass Selectbox & Dropdown Popup List */
    div[data-baseweb="select"] > div {
        background-color: rgba(22, 25, 24, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 16px !important;
        color: #f8fafc !important;
    }
    div[data-baseweb="popover"], ul[data-baseweb="menu"] {
        background-color: rgba(24, 27, 26, 0.92) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 16px !important;
    }
    ul[data-baseweb="menu"] li {
        color: #e2e8f0 !important;
        background-color: transparent !important;
    }
    ul[data-baseweb="menu"] li:hover {
        background-color: rgba(212, 243, 74, 0.18) !important;
        color: #d4f34a !important;
    }

    /* TextArea & TextInput */
    textarea, input {
        background-color: rgba(20, 22, 21, 0.82) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 16px !important;
    }

    /* Sync Button */
    .stButton > button {
        background: #d4f34a !important;
        color: #111312 !important;
        border: none !important;
        border-radius: 30px !important;
        padding: 0.45rem 1.3rem !important;
        font-weight: 700 !important;
    }

    /* Pill Action Links */
    .action-btn-gmail {
        display: inline-block;
        background: #d4f34a;
        color: #111312 !important;
        font-weight: 700;
        padding: 9px 18px;
        border-radius: 30px;
        text-decoration: none !important;
        margin-right: 10px;
        margin-top: 8px;
        font-size: 0.86rem;
    }
    .action-btn-wa {
        display: inline-block;
        background: #4ce0b3;
        color: #111312 !important;
        font-weight: 700;
        padding: 9px 18px;
        border-radius: 30px;
        text-decoration: none !important;
        margin-top: 8px;
        font-size: 0.86rem;
    }
</style>
""", unsafe_allow_html=True)

# 2. Config
SHEET_ID = "1oZZ3L780Bx21aUP-nrb3d33_C9cQwsMd_mqn6AgK26g"
LOCAL_EXCEL_FILE = "Shopx Store Data.xlsx"


# 3. Data Loader
@st.cache_data(ttl=60)
def load_live_data(sheet_id):
    df = None
    source_used = "Local Excel Backup"

    if sheet_id:
        try:
            url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text), header=None)
            source_used = "Live Google Sheet"
        except Exception:
            pass

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


def parse_price(val):
    if pd.isna(val) or str(val).strip() in ['—', '-', '']:
        return np.nan
    cleaned = re.sub(r'[₹,\s]', '', str(val))
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


# 5. ML + SHAP Pipeline
@st.cache_data(ttl=60)
def run_ai_pipeline(df):
    df = df.copy()
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
        return email_val, phone_val

    extracted = cart_df['Contact_Info'].apply(extract_contact_details)
    cart_df['Extracted_Email'] = [e[0] for e in extracted]
    cart_df['Extracted_Phone'] = [e[1] for e in extracted]

    ua_map = (
        df[df['Device_Browser'].astype(str) != 'Web Browser']
        .dropna(subset=['Device_Browser'])
        .groupby('User_Name')['Device_Browser']
        .first()
        .to_dict()
    )
    cart_df['Is_Mobile'] = (
        cart_df['User_Name']
        .map(ua_map)
        .fillna('')
        .astype(str)
        .str.contains('Mobile|Android|iPhone', case=False)
        .astype(int)
    )

    email_to_user = (
        df[df['Event_Type'] == 'SIGNUP']
        .dropna(subset=['Email'])
        .set_index('Email')['User_Name']
        .to_dict()
    )
    failed_df = df[df['Event_Type'] == 'LOGIN_FAILED'].copy()
    failed_df['Mapped_User'] = failed_df['Email'].map(email_to_user).fillna(failed_df['User_Name'])
    friction_map = failed_df.groupby('Mapped_User').size().to_dict()

    clean_df = cart_df[cart_df['User_Name'] != 'Sam'].copy()

    clean_df['Auth_Friction_Count'] = clean_df['User_Name'].map(friction_map).fillna(0)
    clean_df['Cart_Item_Count'] = clean_df.groupby('User_Name')['Product'].transform('count')
    clean_df['User_Total_Cart_Value'] = clean_df.groupby('User_Name')['Price'].transform('sum').replace(0, 1)
    clean_df['Price_Share_In_Cart'] = clean_df['Price'] / clean_df['User_Total_Cart_Value']
    clean_df['Is_High_Ticket'] = (clean_df['Price'] >= 50000).astype(int)
    clean_df['Hour_Of_Day'] = clean_df['Timestamp'].dt.hour

    features = [
        'Price', 'Auth_Friction_Count', 'Cart_Item_Count',
        'User_Total_Cart_Value', 'Price_Share_In_Cart',
        'Is_High_Ticket', 'Is_Mobile', 'Hour_Of_Day'
    ]
    X = clean_df[features].fillna(0)
    y = clean_df['Purchased']

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X, y)

    probs = model.predict_proba(X)
    clean_df['Abandon_Risk_%'] = np.round(probs[:, 0] * 100, 1)

    if SHAP_AVAILABLE:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            abandon_shap = shap_vals[0]
        elif len(shap_vals.shape) == 3:
            abandon_shap = shap_vals[:, :, 0]
        else:
            abandon_shap = shap_vals
        top_driver_idx = np.argmax(abandon_shap, axis=1)
        clean_df['Top_SHAP_Factor'] = [features[i] for i in top_driver_idx]
    else:
        clean_df['Top_SHAP_Factor'] = np.where(
            clean_df['Is_High_Ticket'] == 1, 'Is_High_Ticket',
            np.where(clean_df['Auth_Friction_Count'] > 0, 'Auth_Friction_Count',
            np.where(clean_df['Cart_Item_Count'] > 2, 'Cart_Item_Count', 'Price'))
        )

    reason_labels = {
        'Price': 'High Item Price (Waiting for discount)',
        'User_Total_Cart_Value': 'High Total Cart Value (Comparison shopping)',
        'Cart_Item_Count': 'Multiple Items Added (Product indecision)',
        'Price_Share_In_Cart': 'Expensive Item Dominating Cart',
        'Is_High_Ticket': 'Premium Product (>50k) — Needs EMI/Trust',
        'Auth_Friction_Count': 'Login/Signup Friction Before Checkout',
        'Is_Mobile': 'Mobile Checkout Friction',
        'Hour_Of_Day': 'Off-Peak Window Shopping'
    }
    clean_df['AI_Root_Cause'] = clean_df['Top_SHAP_Factor'].map(reason_labels).fillna('General Hesitation')
    return clean_df


# 6. Personalized Recovery Action Generator
def generate_recovery_action(row):
    user = row['User_Name']
    prod = row['Product']
    price = row['Price'] if pd.notna(row['Price']) else 0
    factor = row['Top_SHAP_Factor']

    base_url = "https://shopx-6zwe.onrender.com/"
    prod_slug = re.sub(r'[^a-zA-Z0-9]+', '-', str(prod).lower()).strip('-')
    cart_url = f"{base_url}?product={prod_slug}"

    if factor in ['Price', 'Is_High_Ticket', 'Price_Share_In_Cart']:
        strategy = "No-Cost EMI + ₹1,500 Instant Discount"
        offer_line = f"Unlock 0% No-Cost EMI + ₹1,500 OFF on {prod} (₹{price:,.0f}) using code SHOPX1500 for the next 4 hours!"
    elif factor in ['User_Total_Cart_Value', 'Cart_Item_Count']:
        strategy = "Bundle Deal (10% Off Multi-Item Cart)"
        offer_line = f"Complete your order for {prod} today and get 10% Bundle Savings (BUNDLE10) + Free Express Delivery!"
    elif factor == 'Auth_Friction_Count':
        strategy = "1-Click Guest Checkout Link"
        offer_line = f"We noticed login trouble earlier while grabbing {prod} — use our 1-Click Guest Checkout to order in 15 seconds!"
    elif factor == 'Is_Mobile':
        strategy = "Mobile 1-Tap UPI + Free Shipping"
        offer_line = f"Pay seamlessly via 1-Tap UPI on mobile for {prod} and get Free Priority Delivery today!"
    else:
        strategy = "Low-Stock Urgency Reminder"
        offer_line = f"{prod} (₹{price:,.0f}) is selling fast — grab yours before stock runs out!"

    msg = (
        f"Hi {user},\n"
        f"You just left behind an item in your shopping cart.\n"
        f"Don't miss the chance to get it!\n\n"
        f"Your cart is missing you: {cart_url}\n"
        f"{offer_line}\n\n"
        f"If you face any issues, feel free to reach out to us with any queries.\n"
        f"ShopX Ecommerce"
    )
    return strategy, msg


# 7. Website Traffic & Conversion Analytics Engine
def compute_traffic_analytics(df):
    total_events = len(df)
    
    visit_events = df[df['Event_Type'].astype(str).str.upper().isin([
        'PAGE_VIEW', 'VISIT', 'SITE_VISIT', 'VIEW_PRODUCT', 'LOGIN', 'SIGNUP'
    ])]
    total_visits = len(visit_events) if not visit_events.empty else total_events
    
    unique_visitors = df['User_Name'].dropna().nunique()
    cart_additions = len(df[df['Event_Type'] == 'ADD_TO_CART'])
    orders_df = df[df['Event_Type'] == 'ORDER_PLACED'].copy()
    total_purchases = len(orders_df)
    
    orders_df['Order_Val'] = orders_df['Price_Raw'].apply(parse_price)
    total_sales_rev = orders_df['Order_Val'].sum() if not orders_df.empty else 0.0
    
    visit_to_buy_rate = (total_purchases / total_visits * 100) if total_visits > 0 else 0.0
    
    traffic_df = df.copy()
    traffic_df['Date'] = traffic_df['Timestamp'].dt.date
    daily_traffic = traffic_df.groupby('Date').size().rename("Website Hits")
    
    funnel_data = pd.DataFrame({
        'Stage': ['1. Total Visits', '2. Added to Cart', '3. Purchased'],
        'Count': [total_visits, cart_additions, total_purchases]
    }).set_index('Stage')
    
    device_series = df['Device_Browser'].fillna('Desktop').astype(str).apply(
        lambda x: 'Mobile' if any(k in x.lower() for k in ['mobile', 'android', 'iphone']) else 'Desktop / Web'
    ).value_counts()

    return {
        'total_visits': total_visits,
        'unique_visitors': unique_visitors,
        'total_purchases': total_purchases,
        'total_sales_rev': total_sales_rev,
        'visit_to_buy_rate': visit_to_buy_rate,
        'daily_traffic': daily_traffic,
        'funnel_data': funnel_data,
        'device_series': device_series
    }


# ================= ORGANIZED DRIBBBLE BENTO GRID UI =================
try:
    raw_df, data_source = load_live_data(SHEET_ID)

    # Top Floating Glass Pill Navbar
    st.markdown(f"""
    <div class="bento-navbar">
        <div class="nav-brand">🛒 ShopX AI Portal</div>
        <div class="nav-pills">
            <span class="nav-pill-active">Dashboard</span>
            <span class="nav-pill">Source: {data_source}</span>
            <span class="nav-pill">Events: {len(raw_df)}</span>
            <span class="nav-pill">Random Forest + SHAP XAI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 Sync Live Data"):
        st.cache_data.clear()
        st.rerun()

    analyzed_df = run_ai_pipeline(raw_df)

    if analyzed_df.empty:
        st.warning("Abhi sheet mein koi ADD_TO_CART event nahi mila.")
        st.stop()

    abandoned_df = (
        analyzed_df[analyzed_df['Purchased'] == 0]
        .sort_values(by='Timestamp', ascending=False)
        .copy()
    )

    recovery_results = abandoned_df.apply(generate_recovery_action, axis=1)
    abandoned_df['Recommended_Action'] = [r[0] for r in recovery_results]
    abandoned_df['Recovery_Message'] = [r[1] for r in recovery_results]

    # Asymmetric Bento Grid: Left Analytics Zone (1.45) + Right Recovery Engine Panel (1)
    left_bento, right_bento = st.columns([1.45, 1], gap="large")

    with left_bento:
        # 1. KPI Summary Strip
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Cart Additions", len(analyzed_df))
        col2.metric("Abandoned", len(abandoned_df))
        col3.metric("Drop-Off Rate", f"{(len(abandoned_df) / len(analyzed_df)) * 100:.1f}%")
        at_risk_millions = abandoned_df['Price'].sum() / 1_000_000
        col4.metric("At-Risk Value", f"₹{at_risk_millions:.2f}M")

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        # 2. Side-by-Side Analytics Bento Cards
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.subheader("📊 Drop-Off Root Causes (SHAP)")
                st.bar_chart(abandoned_df['AI_Root_Cause'].value_counts(), color="#d4f34a")

        with c2:
            with st.container(border=True):
                st.subheader("🔥 Most Abandoned Products")
                short_prods = abandoned_df['Product'].str.slice(0, 20)
                st.bar_chart(short_prods.value_counts().head(6), color="#4ce0b3")

    with right_bento:
        # 3. AI Cart-Recovery & Customer Details Bento Panel
        with st.container(border=True):
            st.subheader("🚀 AI Cart-Recovery Action Engine")
            abandoned_df['Selector_Label'] = (
                abandoned_df['User_Name'].astype(str) + " — " +
                abandoned_df['Product'].str.slice(0, 32) +
                " (₹" + abandoned_df['Price'].fillna(0).astype(int).astype(str) + ")"
            )

            selected_label = st.selectbox(
                "Select an Abandoned Cart Entry:",
                abandoned_df['Selector_Label'].tolist()
            )
            selected_row = abandoned_df[abandoned_df['Selector_Label'] == selected_label].iloc[0]

            # Custom Translucent Glass Slab for Customer Details
            st.markdown(f"""
            <div class="customer-glass-slab">
                <span class="slab-badge-lime">Abandon Risk: {selected_row['Abandon_Risk_%']}%</span>
                <span class="slab-badge-teal">Customer Details</span><br>
                <b>👤 User:</b> {selected_row['User_Name']}<br>
                <b>📞 Contact:</b> {selected_row['Contact_Info']}<br>
                <b>🧠 SHAP Root Cause:</b> {selected_row['AI_Root_Cause']}<br>
                <b>🎯 Strategy:</b> {selected_row['Recommended_Action']}
            </div>
            """, unsafe_allow_html=True)

            edited_msg = st.text_area(
                "Personalized Recovery Message (Editable):",
                value=selected_row['Recovery_Message'],
                height=165
            )
            encoded_text = urllib.parse.quote(edited_msg)
            subject = urllib.parse.quote(f"Complete your ShopX order: {selected_row['Product'][:30]}")

            user_email = str(selected_row.get('Extracted_Email', '')).strip()
            user_phone = str(selected_row.get('Extracted_Phone', '')).strip()

            if not user_phone:
                manual_num = st.text_input(
                    "📱 Enter 10-digit WhatsApp number (if not captured in sheet):",
                    placeholder="e.g. 9876543210"
                )
                clean_manual = re.sub(r'\D', '', manual_num)
                if len(clean_manual) == 10:
                    user_phone = f"91{clean_manual}"
                elif len(clean_manual) == 12 and clean_manual.startswith("91"):
                    user_phone = clean_manual

            wa_link = f"https://wa.me/{user_phone}?text={encoded_text}" if user_phone else f"https://wa.me/?text={encoded_text}"

            buttons_html = ""
            if user_email:
                gmail_link = f"https://mail.google.com/mail/?view=cm&fs=1&to={urllib.parse.quote(user_email)}&su={subject}&body={encoded_text}"
                buttons_html += f'<a href="{gmail_link}" target="_blank" class="action-btn-gmail">📧 Send via Gmail ({user_email})</a>'

            if user_phone:
                buttons_html += f'<a href="{wa_link}" target="_blank" class="action-btn-wa">💬 Direct WhatsApp (+{user_phone})</a>'
            else:
                buttons_html += f'<a href="{wa_link}" target="_blank" class="action-btn-wa">💬 Open WhatsApp</a>'

            st.markdown(buttons_html, unsafe_allow_html=True)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # 4. Website Traffic & Conversion Funnel Bento Section
    traffic = compute_traffic_analytics(raw_df)

    with st.container(border=True):
        st.subheader("🌐 Website Traffic & Purchase Conversion Overview")
        
        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        t_col1.metric("Total Website Visits", f"{traffic['total_visits']:,}", f"{traffic['unique_visitors']} Unique Users")
        t_col2.metric("Completed Purchases", f"{traffic['total_purchases']:,}", f"₹{traffic['total_sales_rev']:,.0f} Revenue")
        t_col3.metric("Visit-to-Buy Conversion", f"{traffic['visit_to_buy_rate']:.1f}%")
        top_device = traffic['device_series'].index[0] if not traffic['device_series'].empty else "Web"
        t_col4.metric("Primary Traffic Source", top_device)

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        tc1, tc2, tc3 = st.columns([1.3, 1, 0.8])
        with tc1:
            st.markdown("**📈 Daily Website Traffic Trend**")
            st.area_chart(traffic['daily_traffic'], color="#d4f34a")
        with tc2:
            st.markdown("**🎯 Visitor-to-Purchase Funnel**")
            st.bar_chart(traffic['funnel_data'], color="#4ce0b3")
        with tc3:
            st.markdown("**📱 Device Traffic Split**")
            st.bar_chart(traffic['device_series'], color="#f5b041")

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # 5. Bottom Full-Width Glass Bento Table
    with st.container(border=True):
        st.subheader("🕵️ Complete Abandonment Diagnosis & Action Table")
        st.dataframe(
            abandoned_df[[
                'Timestamp', 'User_Name', 'Contact_Info', 'Product',
                'Price', 'Abandon_Risk_%', 'AI_Root_Cause', 'Recommended_Action'
            ]],
            use_container_width=True
        )

except Exception as main_err:
    st.error("App run karte waqt ek error aaya:")
    st.exception(main_err)
