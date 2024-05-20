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
clf = joblib.load("clfnew.pkl")
csv_path = 'Twotypes_smote.csv'
df = pd.read_csv(csv_path)
sample_size = 100
indices = np.random.choice(df.index, size=sample_size, replace=False)
X_background = df.drop(['Progression', 'Gender'], axis=1).loc[indices]

# Initialize the explainer with the background data
wrapped_model = StackingClassifierWrapper(clf)
explainer = shap.Explainer(wrapped_model.predict, X_background)

st.header("Streamlit Machine Learning App")
SE_right = st.number_input("SE_right")
School = st.number_input("School")
Vision_correction = st.number_input("Vision_correction")
Age_group = st.number_input("Age_group")
UDVA_group = st.number_input("UDVA_group")

if st.button("Submit"):
    X = pd.DataFrame([[SE_right, School, Vision_correction, Age_group, UDVA_group]], 
                     columns=["SE_right", "School", "Vision_correction", "Age_group", "UDVA_group"])
    
    shap_values = explainer(X)
    rounded_values = np.round(shap_values.values, 2)

    # Attempt to calculate an average base value (use cautiously)
    base_value = np.mean(clf.predict_proba(X_background)[:, 1])  # For binary classification, adjust accordingly

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
