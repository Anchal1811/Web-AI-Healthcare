import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
from statistics import mode

# Load Data
data = pd.read_csv('data/improved_disease_dataset (1).csv')
# Clean column names once at startup
data.columns = [col.strip().lower().replace(" ", "_") for col in data.columns]

encoder = LabelEncoder()
data["disease"] = encoder.fit_transform(data["disease"])

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
symptoms_list = X.columns.values

# Handle Imbalance
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Train
rf_model = RandomForestClassifier(random_state=42).fit(X_resampled, y_resampled)
nb_model = GaussianNB().fit(X_resampled, y_resampled)
svm_model = SVC(random_state=42).fit(X_resampled, y_resampled)

symptom_index = {symptom: idx for idx, symptom in enumerate(symptoms_list)}

def get_prediction(input_symptoms: str):
    # Convert "High Fever, Itching" -> ["high_fever", "itching"]
    user_input = [s.strip().lower().replace(" ", "_") for s in input_symptoms.split(",")]
    
    input_vector = [0] * len(symptom_index)
    valid_count = 0
    
    for s in user_input:
        if s in symptom_index:
            input_vector[symptom_index[s]] = 1
            valid_count += 1
            
    if valid_count == 0:
        return {"rf": "N/A", "nb": "N/A", "svm": "N/A", "final": "No valid symptoms recognized"}

    df = pd.DataFrame([input_vector], columns=symptoms_list)
    
    rf_p = encoder.classes_[rf_model.predict(df)[0]]
    nb_p = encoder.classes_[nb_model.predict(df)[0]]
    svm_p = encoder.classes_[svm_model.predict(df)[0]]
    
    try:
        final_p = mode([rf_p, nb_p, svm_p])
    except:
        final_p = rf_p # Random Forest tie-breaker
        
    return {"rf": rf_p, "nb": nb_p, "svm": svm_p, "final": final_p}