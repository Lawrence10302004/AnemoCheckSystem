import joblib
import numpy as np


def xgboost_predict(user_input:list):
    final_model = joblib.load('cbc_anemia_dataset_v2.pkl')
    # Convert to array and reshape
    sample = np.array(user_input).reshape(1, -1)
    label = ['Mild', 'Moderate', 'Normal', 'Severe']
    # Predict
    prediction = final_model.predict(sample)[0]
    probabilities = final_model.predict_proba(sample)
    confidence_scores = probabilities.max(axis=1)[0]  
    predicted_label = label[prediction]

    print(f"\nPredicted Anemia Severity: **{predicted_label}**, {confidence_scores}")

    

    return predicted_label,confidence_scores


if __name__ == "__main__":
    user_input = [56,1,12.5,5.1,14,37.3,78.9,24.9,316,197,45,45.8,8.6,0.2,1.6,0.2]
    xgboost_predict(user_input)