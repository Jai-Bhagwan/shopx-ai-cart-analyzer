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

# 2. Config: Tumhari Live Google Sheet ID
SHEET_ID = "1oZZ3L780Bx21aUP-nrb3d33_C9cQwsMd_mqn6AgK26g"
LOCAL_EXCEL_FILE = "Shopx Store Data.xlsx"


# 3. Data Loader (5-sec timeout + automatic local Excel fallback)
@st.cache_data(ttl=60)
def load_live_data(sheet_id):
    df = None
    source_used = "Local Excel Backup"

    if sheet_id and sheet_id != "1oZZ3L780Bx21aUP-nrb3d33_C9cQwsMd_mqn6AgK26g":
        try:
            url = f"https://docs.google.com/spreadsheets/d/1oZZ3L780Bx21aUP-nrb3d33_C9cQwsMd_mqn6AgK26g/export?format=csv"
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

    # Map User Email/Phone from Auth/Order events for recovery contact
    user_email_map = (
        df[df['Event_Type'].isin(['SIGNUP', 'LOGIN', 'ORDER_PLACED'])]
        .dropna(subset=['Email'])
        .groupby('User_Name')['Email']
        .first()
        .to_dict()
    )

    # Extract clean product names
    cart_df['Product'] = (
        cart_df['Details']
        .astype(str)
        .str.replace(r'^ADDED:\s*', '', regex=True)
        .str.replace(r'\s*\(x\d+\)$', '', regex=True)
        .str.strip()
    )
    cart_df = cart_df[cart_df['Product'].str.lower() != 'nan'].copy()

    # Label Target: Purchased (1) vs Abandoned (0)
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

    # Extract Email & 10-Digit Indian Mobile Number (+91) separately
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

    # Device mapping (Mobile vs Desktop)
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

    # Auth friction mapping (Failed logins before cart)
    email_to_user = (
        df[df['Event_Type'] == 'SIGNUP']
        .dropna(subset=['Email'])
        .set_index('Email')['User_Name']
        .to_dict()
    )
    failed_df = df[df['Event_Type'] == 'LOGIN_FAILED'].copy()
    failed_df['Mapped_User'] = failed_df['Email'].map(email_to_user).fillna(failed_df['User_Name'])
    friction_map = failed_df.groupby('Mapped_User').size().to_dict()

    # Exclude extreme bulk test outlier ('Sam' 40-item test order)
    clean_df = cart_df[cart_df['User_Name'] != 'Sam'].copy()

    # Behavioral & Cart Feature Engineering
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

    # Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
    model.fit(X, y)

    # Predict Abandonment Risk %
    probs = model.predict_proba(X)
    clean_df['Abandon_Risk_%'] = np.round(probs[:, 0] * 100, 1)

    # SHAP Explainable AI for Row-Level Root Cause
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
    cart_url = "https://shorturl.at/hvRY3"

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


# ================= STREAMLIT DASHBOARD UI =================
try:
    st.title("🛒 ShopX AI — Cart Abandonment & Recovery Engine")

    raw_df, data_source = load_live_data(SHEET_ID)
    st.caption(f"Data Source: **{data_source}** ({len(raw_df)} total events) | Model: **Random Forest + SHAP Explainable AI**")

    if st.button("🔄 Refresh Live Data Now"):
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

    # Top KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Cart Additions", len(analyzed_df))
    col2.metric("Abandoned Items", len(abandoned_df))
    col3.metric("Abandonment Rate", f"{(len(abandoned_df) / len(analyzed_df)) * 100:.1f}%")
    col4.metric("Recoverable Revenue", f"₹{abandoned_df['Price'].sum():,.0f}")

    st.divider()

    # Analytics Charts
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📊 Primary Reasons Users Didn't Buy (SHAP)")
        st.bar_chart(abandoned_df['AI_Root_Cause'].value_counts())

    with c2:
        st.subheader("🔥 Most Abandoned Products")
        short_prods = abandoned_df['Product'].str.slice(0, 24)
        st.bar_chart(short_prods.value_counts().head(6))

    st.divider()

    # AI Cart-Recovery Action Engine
    st.subheader("🚀 AI Cart-Recovery Action Engine")
    abandoned_df['Selector_Label'] = (
        abandoned_df['User_Name'].astype(str) + " — " +
        abandoned_df['Product'].str.slice(0, 35) +
        " (₹" + abandoned_df['Price'].fillna(0).astype(int).astype(str) + ")"
    )

    selected_label = st.selectbox("Select an Abandoned Cart Entry:", abandoned_df['Selector_Label'].tolist())
    selected_row = abandoned_df[abandoned_df['Selector_Label'] == selected_label].iloc[0]

    r_col1, r_col2 = st.columns([1, 1.5])
    with r_col1:
        st.info(
            f"**User:** {selected_row['User_Name']}\n\n"
            f"**Contact:** {selected_row['Contact_Info']}\n\n"
            f"**Abandon Risk Score:** {selected_row['Abandon_Risk_%']}%\n\n"
            f"**SHAP Root Cause:** {selected_row['AI_Root_Cause']}\n\n"
            f"**Recommended Strategy:** {selected_row['Recommended_Action']}"
        )

    with r_col2:
        edited_msg = st.text_area(
            "Personalized Recovery Message (Editable):",
            value=selected_row['Recovery_Message'],
            height=130
        )
        encoded_text = urllib.parse.quote(edited_msg)
        subject = urllib.parse.quote(f"Complete your ShopX order: {selected_row['Product'][:30]}")

        user_email = str(selected_row.get('Extracted_Email', '')).strip()
        user_phone = str(selected_row.get('Extracted_Phone', '')).strip()

        # Optional manual override agar phone number sheet mein logged nahi hai
        if not user_phone:
            manual_num = st.text_input(
                "📱 Phone number sheet mein nahi mila — direct WhatsApp ke liye 10-digit number daalein (Optional):",
                placeholder="e.g. 9876543210"
            )
            clean_manual = re.sub(r'\D', '', manual_num)
            if len(clean_manual) == 10:
                user_phone = f"91{clean_manual}"
            elif len(clean_manual) == 12 and clean_manual.startswith("91"):
                user_phone = clean_manual

        wa_link = f"https://wa.me/{user_phone}?text={encoded_text}" if user_phone else f"https://wa.me/?text={encoded_text}"

        action_links = []
        if user_email:
            gmail_link = f"https://mail.google.com/mail/?view=cm&fs=1&to={urllib.parse.quote(user_email)}&su={subject}&body={encoded_text}"
            action_links.append(f"👉 **[📧 Send via Gmail ({user_email})]({gmail_link})**")

        if user_phone:
            action_links.append(f"👉 **[💬 Direct WhatsApp (+{user_phone})]({wa_link})**")
        else:
            action_links.append(f"👉 **[💬 Open WhatsApp (Select Contact Manually)]({wa_link})**")

        st.markdown(" &nbsp;&nbsp;|&nbsp;&nbsp; ".join(action_links))

    st.divider()

    # Complete Diagnosis Table
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
