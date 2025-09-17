import streamlit as st
from utils.db_utils import insert_new_user

st.set_page_config(page_title="New User Entry", layout="wide")
st.title("📝 New Loan Application Form")

with st.form("new_user_form"):
    st.markdown("### 🔐 Identification")
    SK_ID_PREV = st.number_input("SK_ID_PREV", min_value=0)
    SK_ID_CURR = st.number_input("SK_ID_CURR", min_value=0)

    st.markdown("### 💼 Loan Details")
    NAME_CONTRACT_TYPE = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])
    AMT_ANNUITY = st.number_input("Annuity Amount")
    AMT_APPLICATION = st.number_input("Application Amount")
    AMT_CREDIT = st.number_input("Credit Amount")
    AMT_DOWN_PAYMENT = st.number_input("Down Payment")
    AMT_GOODS_PRICE = st.number_input("Goods Price")

    st.markdown("### 🕒 Application Timing")
    WEEKDAY_APPR_PROCESS_START = st.selectbox("Weekday of Application", 
        ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"])
    HOUR_APPR_PROCESS_START = st.slider("Hour of Application", 0, 23, 10)

    st.markdown("### 📑 Contract Flags")
    FLAG_LAST_APPL_PER_CONTRACT = st.selectbox("Last Application Per Contract", ["Y", "N"])
    NFLAG_LAST_APPL_IN_DAY = st.selectbox("Last Application In Day", [1, 0])
    RATE_DOWN_PAYMENT = st.number_input("Rate of Down Payment")

    st.markdown("### 💳 Loan Purpose & Status")
    NAME_CASH_LOAN_PURPOSE = st.selectbox("Cash Loan Purpose", 
        ["Car purchase", "Repairs", "Business", "Journey", "Other", "Radio/TV", "Medicine"])
    NAME_CONTRACT_STATUS = st.selectbox("Contract Status", ["Approved", "Canceled", "Refused"])
    DAYS_DECISION = st.number_input("Days Decision")

    st.markdown("### 💰 Payment & Rejection")
    NAME_PAYMENT_TYPE = st.selectbox("Payment Type", ["Cash through the bank", "Non-cash"])
    CODE_REJECT_REASON = st.selectbox("Reject Reason Code", ["HC", "LIMIT", "SCO"])
    
    st.markdown("### 👨‍👩‍👧‍👦 Client Info")
    NAME_TYPE_SUITE = st.selectbox("Type Suite", ["Unaccompanied", "Family", "Spouse", "Group"])
    NAME_CLIENT_TYPE = st.selectbox("Client Type", ["New", "Repeater"])
    
    st.markdown("### 🛍️ Goods Info")
    NAME_GOODS_CATEGORY = st.selectbox("Goods Category", ["Furniture", "Consumer Electronics", "Computers"])
    NAME_PORTFOLIO = st.selectbox("Portfolio", ["POS", "Cards"])
    NAME_PRODUCT_TYPE = st.selectbox("Product Type", ["X-Sell", "Walk-in"])
    
    st.markdown("### 🌐 Channel & Seller Info")
    CHANNEL_TYPE = st.selectbox("Channel Type", ["Country-wide", "Credit and cash offices"])
    SELLERPLACE_AREA = st.number_input("Seller Place Area")
    NAME_SELLER_INDUSTRY = st.selectbox("Seller Industry", ["Industry type 1", "Industry type 2", "Trade", "Transport"])

    st.markdown("### 🧾 Payment Plan")
    CNT_PAYMENT = st.number_input("Count of Payments")
    NAME_YIELD_GROUP = st.selectbox("Yield Group", ["high", "low", "middle", "XNA"])
    PRODUCT_COMBINATION = st.text_input("Product Combination", "XNA")

    st.markdown("### 📅 Timeline of Repayment")
    DAYS_FIRST_DRAWING = st.number_input("Days First Drawing")
    DAYS_FIRST_DUE = st.number_input("Days First Due")
    DAYS_LAST_DUE_1ST_VERSION = st.number_input("Days Last Due (1st Version)")
    DAYS_LAST_DUE = st.number_input("Days Last Due")
    DAYS_TERMINATION = st.number_input("Days Termination")
    NFLAG_INSURED_ON_APPROVAL = st.selectbox("Insured on Approval", [0, 1])

    submitted = st.form_submit_button("🚀 Submit Application")

    if submitted:
        data = (
            SK_ID_PREV, SK_ID_CURR, NAME_CONTRACT_TYPE, AMT_ANNUITY, AMT_APPLICATION,
            AMT_CREDIT, AMT_DOWN_PAYMENT, AMT_GOODS_PRICE, WEEKDAY_APPR_PROCESS_START,
            HOUR_APPR_PROCESS_START, FLAG_LAST_APPL_PER_CONTRACT, NFLAG_LAST_APPL_IN_DAY,
            RATE_DOWN_PAYMENT, NAME_CASH_LOAN_PURPOSE, NAME_CONTRACT_STATUS, DAYS_DECISION,
            NAME_PAYMENT_TYPE, CODE_REJECT_REASON, NAME_TYPE_SUITE, NAME_CLIENT_TYPE,
            NAME_GOODS_CATEGORY, NAME_PORTFOLIO, NAME_PRODUCT_TYPE, CHANNEL_TYPE,
            SELLERPLACE_AREA, NAME_SELLER_INDUSTRY, CNT_PAYMENT, NAME_YIELD_GROUP,
            PRODUCT_COMBINATION, DAYS_FIRST_DRAWING, DAYS_FIRST_DUE, DAYS_LAST_DUE_1ST_VERSION,
            DAYS_LAST_DUE, DAYS_TERMINATION, NFLAG_INSURED_ON_APPROVAL
        )
        success = insert_new_user(data)
        if success:
            st.success("✅ New user data successfully inserted into the database!")
        else:
            st.error("❌ Error inserting data. Please try again.")

# pages/New_User.py

# import streamlit as st
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from utils.db_utils import insert_new_user

# st.set_page_config(page_title="New User Entry", layout="wide")
# st.title("📝 New Loan Application Form")

# with st.form("new_user_form"):
#     st.markdown("### 🔐 Identification")
#     SK_ID_PREV = st.number_input("SK_ID_PREV", min_value=0)
#     SK_ID_CURR = st.number_input("SK_ID_CURR", min_value=0)

#     st.markdown("### 💼 Loan Details")
#     NAME_CONTRACT_TYPE = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])
#     AMT_ANNUITY = st.number_input("Annuity Amount")
#     AMT_APPLICATION = st.number_input("Application Amount")
#     AMT_CREDIT = st.number_input("Credit Amount")
#     AMT_DOWN_PAYMENT = st.number_input("Down Payment")
#     AMT_GOODS_PRICE = st.number_input("Goods Price")

#     st.markdown("### 🕒 Application Timing")
#     WEEKDAY_APPR_PROCESS_START = st.selectbox("Weekday of Application", 
#         ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"])
#     HOUR_APPR_PROCESS_START = st.slider("Hour of Application", 0, 23, 10)

#     st.markdown("### 📑 Contract Flags")
#     FLAG_LAST_APPL_PER_CONTRACT = st.selectbox("Last Application Per Contract", ["Y", "N"])
#     NFLAG_LAST_APPL_IN_DAY = st.selectbox("Last Application In Day", [1, 0])
#     RATE_DOWN_PAYMENT = st.number_input("Rate of Down Payment")

#     st.markdown("### 💳 Loan Purpose & Status")
#     NAME_CASH_LOAN_PURPOSE = st.selectbox("Cash Loan Purpose", 
#         ["Car purchase", "Repairs", "Business", "Journey", "Other", "Radio/TV", "Medicine"])
#     NAME_CONTRACT_STATUS = st.selectbox("Contract Status", ["Approved", "Canceled", "Refused"])
#     DAYS_DECISION = st.number_input("Days Decision")

#     st.markdown("### 💰 Payment & Rejection")
#     NAME_PAYMENT_TYPE = st.selectbox("Payment Type", ["Cash through the bank", "Non-cash"])
#     CODE_REJECT_REASON = st.selectbox("Reject Reason Code", ["HC", "LIMIT", "SCO"])
    
#     st.markdown("### 👨‍👩‍👧‍👦 Client Info")
#     NAME_TYPE_SUITE = st.selectbox("Type Suite", ["Unaccompanied", "Family", "Spouse", "Group"])
#     NAME_CLIENT_TYPE = st.selectbox("Client Type", ["New", "Repeater"])
    
#     st.markdown("### 🛍️ Goods Info")
#     NAME_GOODS_CATEGORY = st.selectbox("Goods Category", ["Furniture", "Consumer Electronics", "Computers"])
#     NAME_PORTFOLIO = st.selectbox("Portfolio", ["POS", "Cards"])
#     NAME_PRODUCT_TYPE = st.selectbox("Product Type", ["X-Sell", "Walk-in"])
    
#     st.markdown("### 🌐 Channel & Seller Info")
#     CHANNEL_TYPE = st.selectbox("Channel Type", ["Country-wide", "Credit and cash offices"])
#     SELLERPLACE_AREA = st.number_input("Seller Place Area")
#     NAME_SELLER_INDUSTRY = st.selectbox("Seller Industry", ["Industry type 1", "Industry type 2", "Trade", "Transport"])

#     st.markdown("### 🧾 Payment Plan")
#     CNT_PAYMENT = st.number_input("Count of Payments")
#     NAME_YIELD_GROUP = st.selectbox("Yield Group", ["high", "low", "middle", "XNA"])
#     PRODUCT_COMBINATION = st.text_input("Product Combination", "XNA")

#     st.markdown("### 📅 Timeline of Repayment")
#     DAYS_FIRST_DRAWING = st.number_input("Days First Drawing")
#     DAYS_FIRST_DUE = st.number_input("Days First Due")
#     DAYS_LAST_DUE_1ST_VERSION = st.number_input("Days Last Due (1st Version)")
#     DAYS_LAST_DUE = st.number_input("Days Last Due")
#     DAYS_TERMINATION = st.number_input("Days Termination")
#     NFLAG_INSURED_ON_APPROVAL = st.selectbox("Insured on Approval", [0, 1])

#     submitted = st.form_submit_button("🚀 Submit Application")

#     if submitted:
#         data = (
#             SK_ID_PREV, SK_ID_CURR, NAME_CONTRACT_TYPE, AMT_ANNUITY, AMT_APPLICATION,
#             AMT_CREDIT, AMT_DOWN_PAYMENT, AMT_GOODS_PRICE, WEEKDAY_APPR_PROCESS_START,
#             HOUR_APPR_PROCESS_START, FLAG_LAST_APPL_PER_CONTRACT, NFLAG_LAST_APPL_IN_DAY,
#             RATE_DOWN_PAYMENT, NAME_CASH_LOAN_PURPOSE, NAME_CONTRACT_STATUS, DAYS_DECISION,
#             NAME_PAYMENT_TYPE, CODE_REJECT_REASON, NAME_TYPE_SUITE, NAME_CLIENT_TYPE,
#             NAME_GOODS_CATEGORY, NAME_PORTFOLIO, NAME_PRODUCT_TYPE, CHANNEL_TYPE,
#             SELLERPLACE_AREA, NAME_SELLER_INDUSTRY, CNT_PAYMENT, NAME_YIELD_GROUP,
#             PRODUCT_COMBINATION, DAYS_FIRST_DRAWING, DAYS_FIRST_DUE, DAYS_LAST_DUE_1ST_VERSION,
#             DAYS_LAST_DUE, DAYS_TERMINATION, NFLAG_INSURED_ON_APPROVAL
#         )
#         success = insert_new_user(data)
#         if success:
#             st.success("✅ New user data successfully inserted into the database!")
#         else:
#             st.error("❌ Error inserting data. Please try again.")

