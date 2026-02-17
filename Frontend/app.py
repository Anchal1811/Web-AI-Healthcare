import streamlit as st
import requests

st.set_page_config(page_title="Health AI", layout="wide")
st.title("AI Disease Prediction System")

# 1. This list MUST match your CSV columns exactly
symptom_options = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", 
    "shivering", "chills", "joint_pain", "stomach_pain", "acidity", 
    "ulcers_on_tongue", "vomiting", "fatigue", "weight_loss", "restlessness", 
    "lethargy", "cough", "high_fever", "sunken_eyes", "breathlessness", "sweating"
]

st.markdown("### Select Symptoms")
# 2. Use a multiselect so users pick exact terms
selected = st.multiselect("Pick symptoms the patient is experiencing:", symptom_options)

if st.button("Analyze Health Data"):
    if selected:
        # Convert list to comma-separated string for your Backend
        symptoms_string = ",".join(selected)
        
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict", 
                json={"symptoms": symptoms_string}
            )
            
            if response.status_code == 200:
                result = response.json()["predictions"]
                st.info(f"### Final Diagnosis: **{result['Final Prediction']}**")
                
                # Show model confidence
                with st.expander("See Detailed Model Breakdown"):
                    st.write(f"Random Forest: {result['Random Forest Prediction']}")
                    st.write(f"SVM: {result['SVM Prediction']}")
                    st.write(f"Naive Bayes: {result['Naive Bayes Prediction']}")
            else:
                st.error("Backend logic error.")
        except:
            st.error("Backend Connection Failed. Check Terminal 1.")
    else:
        st.warning("Please select at least 2-3 symptoms for an accurate reading.")