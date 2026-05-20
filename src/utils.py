import shap
import numpy as np

def generate_clinical_explanation(model, patient_tensor, background_data, feature_names):
    explainer = shap.DeepExplainer(model, background_data)
    raw_shap_values = explainer.shap_values(patient_tensor)
    if isinstance(raw_shap_values, list): raw_shap_values = raw_shap_values[0]
        
    aggregated_shap = np.sum(raw_shap_values, axis=1)
    mean_patient_vitals = np.mean(patient_tensor[0], axis=0)
    
    explanation = shap.Explanation(
        values=aggregated_shap[0], 
        base_values=explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value, 
        data=mean_patient_vitals,
        feature_names=feature_names
    )
    return explanation 