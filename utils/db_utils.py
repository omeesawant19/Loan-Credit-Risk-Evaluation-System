import mysql.connector
import streamlit as st

def get_connection():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            port=3306,
            user="root",
            password="rushi@492001#",
            database="loan_dataset"
        )
        return conn
    except mysql.connector.Error as err:
        st.error(f"❌ MySQL Error: {err}")
        return None

def fetch_existing_user(sk_id_prev, sk_id_curr):
    conn = get_connection()
    if conn:
        cursor = conn.cursor(dictionary=True)
        query = """
        SELECT * FROM cleaned_data_final
        WHERE SK_ID_PREV = %s AND SK_ID_CURR = %s
        """
        cursor.execute(query, (sk_id_prev, sk_id_curr))
        result = cursor.fetchall()
        cursor.close()
        conn.close()
        return result
    return []

def insert_new_user(data):
    conn = get_connection()
    if conn:
        cursor = conn.cursor()
        query = """
        INSERT INTO cleaned_data_final (
            SK_ID_PREV, SK_ID_CURR, NAME_CONTRACT_TYPE, AMT_ANNUITY, AMT_APPLICATION,
            AMT_CREDIT, AMT_DOWN_PAYMENT, AMT_GOODS_PRICE, WEEKDAY_APPR_PROCESS_START,
            HOUR_APPR_PROCESS_START, FLAG_LAST_APPL_PER_CONTRACT, NFLAG_LAST_APPL_IN_DAY,
            RATE_DOWN_PAYMENT, NAME_CASH_LOAN_PURPOSE, NAME_CONTRACT_STATUS, DAYS_DECISION,
            NAME_PAYMENT_TYPE, CODE_REJECT_REASON, NAME_TYPE_SUITE, NAME_CLIENT_TYPE,
            NAME_GOODS_CATEGORY, NAME_PORTFOLIO, NAME_PRODUCT_TYPE, CHANNEL_TYPE,
            SELLERPLACE_AREA, NAME_SELLER_INDUSTRY, CNT_PAYMENT, NAME_YIELD_GROUP,
            PRODUCT_COMBINATION, DAYS_FIRST_DRAWING, DAYS_FIRST_DUE, DAYS_LAST_DUE_1ST_VERSION,
            DAYS_LAST_DUE, DAYS_TERMINATION, NFLAG_INSURED_ON_APPROVAL
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, data)
        conn.commit()
        cursor.close()
        conn.close()
        return True
    return False



# from utils.db_connection import get_connection

# def insert_new_user(data):
#     conn = get_connection()
#     if conn:
#         try:
#             cursor = conn.cursor()
#             cursor.execute("""
#                 INSERT INTO historical_data (
#                     SK_ID_PREV, SK_ID_CURR, NAME_CONTRACT_TYPE, AMT_ANNUITY, AMT_APPLICATION,
#                     AMT_CREDIT, AMT_DOWN_PAYMENT, AMT_GOODS_PRICE, WEEKDAY_APPR_PROCESS_START,
#                     HOUR_APPR_PROCESS_START, FLAG_LAST_APPL_PER_CONTRACT, NFLAG_LAST_APPL_IN_DAY,
#                     RATE_DOWN_PAYMENT, NAME_CASH_LOAN_PURPOSE, NAME_CONTRACT_STATUS, DAYS_DECISION,
#                     NAME_PAYMENT_TYPE, CODE_REJECT_REASON, NAME_TYPE_SUITE, NAME_CLIENT_TYPE,
#                     NAME_GOODS_CATEGORY, NAME_PORTFOLIO, NAME_PRODUCT_TYPE, CHANNEL_TYPE,
#                     SELLERPLACE_AREA, NAME_SELLER_INDUSTRY, CNT_PAYMENT, NAME_YIELD_GROUP,
#                     PRODUCT_COMBINATION, DAYS_FIRST_DRAWING, DAYS_FIRST_DUE, DAYS_LAST_DUE_1ST_VERSION,
#                     DAYS_LAST_DUE, DAYS_TERMINATION, NFLAG_INSURED_ON_APPROVAL
#                 )
#                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#             """, data)
#             conn.commit()
#             cursor.close()
#             conn.close()
#             return True
#         except Exception as e:
#             print("❌ Insert Error:", e)
#             return False
#     return False


# def fetch_existing_user(sk_id_prev, sk_id_curr):
#     conn = get_connection()
#     if conn:
#         try:
#             cursor = conn.cursor()
#             cursor.execute("""
#                 SELECT * FROM historical_data
#                 WHERE SK_ID_PREV = ? AND SK_ID_CURR = ?
#             """, (sk_id_prev, sk_id_curr))
#             rows = cursor.fetchall()
#             columns = [column[0] for column in cursor.description]
#             data = [dict(zip(columns, row)) for row in rows]
#             cursor.close()
#             conn.close()
#             return data
#         except Exception as e:
#             print("❌ Fetch Error:", e)
#             return []
#     return []

