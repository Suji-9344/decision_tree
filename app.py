import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(page_title="Admission Prediction App")
st.title("🎓 Admission Prediction App")
st.write("Predict whether a student gets admission")

# --------------------------------
# DATASET (EMBEDDED – NO CSV ERROR)
# --------------------------------
data = {
    'GRE_Score': [300, 310, 320, 330, 340, 305, 315, 325, 335, 345],
    'TOEFL_Score': [100, 105, 110, 115, 120, 102, 107, 112, 117, 120],
    'University_Rating': [3, 4, 5, 4, 5, 3, 4, 5, 4, 5],
    'SOP': [3.0, 3.5, 4.0, 4.5, 5.0, 3.2, 3.8, 4.2, 4.8, 5.0],
    'LOR': [3.0, 3.5, 4.0, 4.5, 5.0, 3.2, 3.8, 4.2, 4.8, 5.0],
    'CGPA': [7.0, 7.5, 8.0, 8.5, 9.0, 7.2, 7.8, 8.2, 8.8, 9.2],
    'Research': [0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    'Admission': [0, 0, 1, 1, 1, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# --------------------------------
# SHOW DATASET
# --------------------------------
with st.expander("📊 View Dataset"):
    st.dataframe(df)

# --------------------------------
# DATA PREPARATION
# --------------------------------
X = df.drop('Admission', axis=1)
y = df['Admission']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------
# MODEL TRAINING
# --------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --------------------------------
# MODEL EVALUATION
# --------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📈 Model Performance")
st.write(f"✅ Accuracy: **{accuracy * 100:.2f}%**")

cm = confusion_matrix(y_test, y_pred)
st.write("Confusion Matrix:")
st.write(cm)

# --------------------------------
# USER INPUT
# --------------------------------
st.subheader("🧑‍🎓 Enter Student Details")

gre = st.number_input("GRE Score", 260, 340, 320)
toefl = st.number_input("TOEFL Score", 90, 120, 110)
rating = st.selectbox("University Rating", [1, 2, 3, 4, 5])
sop = st.slider("SOP Strength", 1.0, 5.0, 4.0)
lor = st.slider("LOR Strength", 1.0, 5.0, 4.0)
cgpa = st.slider("CGPA", 6.0, 10.0, 8.5)
research = st.selectbox("Research Experience", [0, 1])

# --------------------------------
# PREDICTION
# --------------------------------
if st.button("Predict Admission"):
    input_df = pd.DataFrame([[gre, toefl, rating, sop, lor, cgpa, research]],
                            columns=X.columns)
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success("🎉 Admission Likely!")
    else:
        st.error("❌ Admission Not Likely")
