import streamlit as st
import mysql.connector
import pandas as pd

# -------------------------------
# ✅ DB CONNECTION
# -------------------------------
def get_connection():
    try:
        return mysql.connector.connect(
            host="localhost",
            port=3306,
            user="root",
            password="rushi@492001#",
            database="loan_dataset"
        )
    except mysql.connector.Error as err:
        st.error(f"❌ MySQL Error: {err}")
        return None

# -------------------------------
# 🔍 FETCH USER DATA
# -------------------------------
def fetch_existing_user(sk_id_prev, sk_id_curr):
    conn = get_connection()
    if conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT * FROM cleaned_data_final
            WHERE SK_ID_PREV = %s AND SK_ID_CURR = %s
        """, (sk_id_prev, sk_id_curr))
        result = cursor.fetchall()
        cursor.close()
        conn.close()
        return result
    return []

# -------------------------------
# 🚀 STREAMLIT PAGE
# -------------------------------
st.set_page_config(page_title="View Existing User", layout="centered")
st.title("🏦 Existing User Lookup")

with st.form("fetch_form"):
    sk_id_prev = st.text_input("Enter SK_ID_PREV")
    sk_id_curr = st.text_input("Enter SK_ID_CURR")
    fetch = st.form_submit_button("🔍 Fetch User")

if fetch:
    if sk_id_prev.strip() and sk_id_curr.strip():
        user_data = fetch_existing_user(sk_id_prev, sk_id_curr)

        if user_data:
            st.success("✅ User Found!")
            st.dataframe(pd.DataFrame(user_data))
        else:
            st.error("❌ No user found with those IDs.")
    else:
        st.error("⚠ Please enter both SK_ID_PREV and SK_ID_CURR.")


# pages/Existing_User.py

# import streamlit as st
# import pandas as pd
# import sys
# import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from utils.db_utils import fetch_existing_user

# st.set_page_config(page_title="View Existing User", layout="centered")
# st.title("🏦 Existing User Lookup")

# with st.form("fetch_form"):
#     sk_id_prev = st.text_input("Enter SK_ID_PREV")
#     sk_id_curr = st.text_input("Enter SK_ID_CURR")
#     fetch = st.form_submit_button("🔍 Fetch User")

# if fetch:
#     if sk_id_prev.isdigit() and sk_id_curr.isdigit():
#         user_data = fetch_existing_user(int(sk_id_prev), int(sk_id_curr))
#         if user_data:
#             st.success("✅ User Found!")
#             st.dataframe(pd.DataFrame(user_data))
#         else:
#             st.error("❌ No user found with those IDs.")
#     else:
#         st.warning("⚠️ Please enter valid numeric IDs.")



