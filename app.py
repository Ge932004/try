import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib

class StackingClassifierWrapper:
    def __init__(self, model):
        self.model = model
    def predict(self, X):
        return self.model.predict(X)  
    def predict_proba(self, X):
        return self.model.predict_proba(X)

# Load the classifier and data outside the button logic
clf = joblib.load("clfnumnew.pkl")
csv_path = 'numTwotypes_smote.csv'
df = pd.read_csv(csv_path)
df_dummies = pd.get_dummies(df[['School','Vision_correction', 'Gender']])
df_new = pd.concat([df.drop(['School', 'Vision_correction', 'Gender'], axis=1), df_dummies], axis=1)
X_all = df_new.drop(['Progression'],axis = 1)

# Initialize the explainer with the background data
wrapped_model = StackingClassifierWrapper(clf)
explainer = shap.Explainer(wrapped_model.predict, X_all)

Min_values = {
    "SE_right": -14.625,
    "Age": 6.088,
    "UDVA": -0.079
}

Max_values = {
    "SE_right": -0.50,
    "Age": 18.101,
    "UDVA": 1.602
}

st.header("Continous Myopia Progression Prediction App")
SE_right = st.number_input("SE_right")
Age = st.number_input("Age")
UDVA = st.number_input("UDVA")
School = st.number_input("School")
Vision_correction = st.number_input("Vision_correction")
Gender =st.number_input("Gender")

if st.button("Submit"):
    SE_right_std = (SE_right - Min_values["SE_right"]) / (Max_values["SE_right"] - Min_values["SE_right"])
    Age_std = (Age - Min_values["Age"]) / (Max_values["Age"]-Min_values["Age"])
    UDVA_std = (Logitech(1/UDVA) - Min_values["UDVA"]) / (Max_values["UDVA"]-Min_values["UDVA"])

    X = pd.DataFrame([[SE_right_std, Age_std, UDVA_std, School, Vision_correction,   Gender]], 
                     columns=["SE_right", "Age", "UDVA","School", "Vision_correction", "Gender"])
     
    shap_values = explainer(X)
    rounded_values = np.round(shap_values.values, 2)

    # Attempt to calculate an average base value (use cautiously)
    base_value = np.mean (clf.predict_proba(X_all)[:, 1])  # For binary classification, adjust accordingly

    # Generate and display SHAP force plot
    shap.force_plot(base_value, rounded_values[0], X.iloc[0],
                    feature_names=X.columns.tolist(), matplotlib=True, show=False)
    plt.xticks(fontproperties='Times New Roman', size=15)
    plt.yticks(fontproperties='Times New Roman', size=20)
    plt.tight_layout()
    plt.savefig("shap_plot.png")
    st.image('shap_plot.png')
    
    pred = wrapped_model.predict_proba(X)
    st.markdown("#### _Based on feature values, predicted possibility of continous myopia progression is {:.2%}_".format(pred[0][1]))
    prediction = wrapped_model.predict(X)[0]
    st.text(f"This instance is a {prediction}")
