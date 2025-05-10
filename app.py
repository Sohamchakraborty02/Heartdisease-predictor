import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
with open("Random_forest_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

st.title("💖 Heart Disease Prediction App")
st.write("This app predicts the likelihood of a patient having heart disease using machine learning.")

# Sidebar for user input
st.sidebar.header("User Input Features")
def user_input_features():
    age = st.sidebar.slider("Age", 29, 71, 50)
    sex = st.sidebar.radio("Sex", ("Female", "Male"))
    cp = st.sidebar.selectbox("Chest Pain Type (CP)", [0, 1, 2, 3])
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 94, 200, 120)
    chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 126, 564, 240)
    fbs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    restecg = st.sidebar.selectbox("Resting ECG Results", [0, 1, 2])
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 71, 202, 150)
    exang = st.sidebar.radio("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment", [1, 2, 3])
    ca = st.sidebar.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thalassemia", [0, 1, 2])

    data = {
        'age': age, 'sex': 1 if sex == "Male" else 0, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': 1 if fbs == "Yes" else 0, 'restecg': restecg,
        'thalach': thalach, 'exang': 1 if exang == "Yes" else 0, 'oldpeak': oldpeak,
        'slope': slope, 'ca': ca, 'thal': thal
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()
st.subheader("User Input Features")
st.write(input_df)

# Load dataset for feature alignment
df = pd.read_csv("heart.csv")
df = df.drop(columns=['target'])
df = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
input_df = pd.get_dummies(input_df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
input_df = input_df.reindex(columns=df.columns, fill_value=0)

# Make predictions
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0]

st.subheader("Prediction Result")
if prediction == 1:
    st.error("❌ High Risk of Heart Disease!")
else:
    st.success("✅ Low Risk of Heart Disease!")

# Show prediction probability
st.subheader("Prediction Probability")
st.write(f"Probability of No Disease: {prediction_proba[0]:.2f}")
st.write(f"Probability of Disease: {prediction_proba[1]:.2f}")

# Visualize Probability as Meter Gauge
st.subheader("Risk Meter")
st.progress(int(prediction_proba[1] * 100))

# Show Feature Importance
st.subheader("Feature Importance")
feature_importances = pd.Series(model.feature_importances_, index=df.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 5))
sns.barplot(x=feature_importances, y=feature_importances.index, palette='coolwarm')
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top Features Contributing to Prediction")
st.pyplot(plt)

st.write("📌 **Note:** This model is based on historical data and provides a probability estimate, not a definitive diagnosis.")
