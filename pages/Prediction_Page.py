import streamlit as st
import pandas as pd
import joblib
from groq import Groq

# ✅ Set Groq API Key
groq_client = Groq(api_key="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")  # Replace with your actual key

# ✅ Page setup
st.set_page_config(page_title="Loan Prediction", layout="wide")
st.title("🔮 Loan Prediction Engine")
st.markdown("Fill in all applicant details or chat with the assistant to auto-fill.")

# ✅ Load model
try:
    model = joblib.load("xgboost_home_loan_model.pkl")
except FileNotFoundError:
    st.error("❌ Model file not found!")
    st.stop()

# ✅ Extract form fields from chat prompt
def extract_fields_from_prompt(prompt):
    schema = """
    Extract and return the following fields as JSON:
    - income: Annual income in INR (float)
    - credit_amt: Loan credit amount (float)
    - children: Number of children (int)
    - days_birth: Age in days (negative int)
    - days_employed: Days employed (negative int)
    - ext_source_1: External Score 1 (float, 0-1)
    - ext_source_2: External Score 2 (float, 0-1)
    - ext_source_3: External Score 3 (float, 0-1)
    - education: One of ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary"]
    - family_status: One of ["Married", "Single / not married", "Civil marriage", "Separated"]
    - contract_type: One of ["Cash loans", "Revolving loans"]
    - gender: "Male" or "Female"
    - occupation: One of ["Core staff", "Laborers", "Accountants", "Managers", "Drivers", "Sales staff", "Others"]
    - region_pop: Region population relative (float 0-1)
    - days_publish: Days since ID published (negative int)
    - own_car: "Yes" or "No"
    """

    prompt_text = f"""
    {schema}
    User input:
    {prompt}
    Return JSON only.
    """

    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt_text}]
    )

    import json
    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {}

# ✅ Chatbot section
if "messages" not in st.session_state:
    st.session_state.messages = []

st.subheader("💬 Loan Application Assistant")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Tell me about yourself and your loan request...")
data = {}

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Extracting info..."):
            data = extract_fields_from_prompt(prompt)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"✅ I extracted these: {data}"
            })
            st.markdown(f"✅ Extracted data: `{data}`")

# ✅ Prediction Form
with st.form("loan_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        income = st.number_input("Annual Income (INR)", value=float(data.get("income", 0.0)), min_value=0.0, step=1000.0)
        credit_amt = st.number_input("Loan Credit Amount", value=float(data.get("credit_amt", 0.0)), min_value=0.0, step=1000.0)
        children = st.number_input("Number of Children", value=int(data.get("children", 0)), min_value=0, max_value=10)
        days_birth = st.number_input("Age in Days (Negative)", value=int(data.get("days_birth", -12000)))
        days_employed = st.number_input("Days Employed (Negative)", value=int(data.get("days_employed", -500)))
        ext_source_1 = st.number_input("External Score 1", value=float(data.get("ext_source_1", 0.5)), min_value=0.0, max_value=1.0, step=0.01)
        ext_source_2 = st.number_input("External Score 2", value=float(data.get("ext_source_2", 0.5)), min_value=0.0, max_value=1.0, step=0.01)
        ext_source_3 = st.number_input("External Score 3", value=float(data.get("ext_source_3", 0.5)), min_value=0.0, max_value=1.0, step=0.01)

    with col2:
        education = st.selectbox("Education Level", [
            "Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary"
        ], index=["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary"].index(data.get("education", "Secondary / secondary special")))

        family_status = st.selectbox("Family Status", [
            "Married", "Single / not married", "Civil marriage", "Separated"
        ], index=["Married", "Single / not married", "Civil marriage", "Separated"].index(data.get("family_status", "Married")))

        contract_type = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"],
            index=["Cash loans", "Revolving loans"].index(data.get("contract_type", "Cash loans")))

        gender = st.selectbox("Gender", ["Male", "Female"],
            index=["Male", "Female"].index(data.get("gender", "Male")))

        occupation = st.selectbox("Occupation Type", [
            "Core staff", "Laborers", "Accountants", "Managers", "Drivers", "Sales staff", "Others"
        ], index=["Core staff", "Laborers", "Accountants", "Managers", "Drivers", "Sales staff", "Others"].index(data.get("occupation", "Core staff")))

        region_pop = st.number_input("Region Population Relative", value=float(data.get("region_pop", 0.01)), min_value=0.0, max_value=1.0, step=0.01)
        days_publish = st.number_input("Days Since ID Published", value=int(data.get("days_publish", -1000)))
        own_car = st.selectbox("Owns a Car?", ["No", "Yes"],
            index=["No", "Yes"].index(data.get("own_car", "No")))

    submit = st.form_submit_button("🚀 Predict")

    if submit:
        # Map categorical features
        edu_map = {
            "Secondary / secondary special": 0, "Higher education": 1,
            "Incomplete higher": 2, "Lower secondary": 3
        }
        fam_map = {
            "Married": 0, "Single / not married": 1,
            "Civil marriage": 2, "Separated": 3
        }
        contract_map = {"Cash loans": 0, "Revolving loans": 1}
        gender_map = {"Male": 0, "Female": 1}
        occupation_map = {
            "Core staff": 0, "Laborers": 1, "Accountants": 2, "Managers": 3,
            "Drivers": 4, "Sales staff": 5, "Others": 6
        }
        car_map = {"No": 0, "Yes": 1}

        input_df = pd.DataFrame({
            "AMT_INCOME_TOTAL": [income],
            "AMT_CREDIT": [credit_amt],
            "CNT_CHILDREN": [children],
            "DAYS_BIRTH": [days_birth],
            "NAME_EDUCATION_TYPE": [edu_map[education]],
            "NAME_FAMILY_STATUS": [fam_map[family_status]],
            "NAME_CONTRACT_TYPE": [contract_map[contract_type]],
            "CODE_GENDER": [gender_map[gender]],
            "DAYS_EMPLOYED": [days_employed],
            "OCCUPATION_TYPE": [occupation_map[occupation]],
            "EXT_SOURCE_1": [ext_source_1],
            "EXT_SOURCE_2": [ext_source_2],
            "EXT_SOURCE_3": [ext_source_3],
            "REGION_POPULATION_RELATIVE": [region_pop],
            "DAYS_ID_PUBLISH": [days_publish],
            "FLAG_OWN_CAR": [car_map[own_car]]
        })

        try:
            prediction = model.predict(input_df)[0]
            result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Not Approved"
            st.subheader("Prediction Result:")
            st.success(result)
        except Exception as e:
            st.error(f"🚫 Prediction failed: {e}")
