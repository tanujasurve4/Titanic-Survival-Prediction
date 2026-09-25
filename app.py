import streamlit as st
import pandas as pd
import joblib


# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)


# -------------------------------------------------
# Load Trained Model
# -------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("titanic_model.pkl")


model = load_model()


# -------------------------------------------------
# Title
# -------------------------------------------------

st.title("🚢 Titanic Survival Prediction")

st.write(
    "This application uses Machine Learning to predict "
    "whether a Titanic passenger would have survived "
    "based on passenger information."
)

st.divider()


# -------------------------------------------------
# Passenger Information
# -------------------------------------------------

st.subheader("👤 Enter Passenger Information")

col1, col2 = st.columns(2)


with col1:

    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

    sex = st.selectbox(
        "Gender",
        ["male", "female"]
    )

    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=1.0
    )

    sibsp = st.number_input(
        "Number of Siblings / Spouses",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )


with col2:

    parch = st.number_input(
        "Number of Parents / Children",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    fare = st.number_input(
        "Fare",
        min_value=0.0,
        max_value=600.0,
        value=32.0,
        step=1.0
    )

    embarked = st.selectbox(
        "Port of Embarkation",
        ["S", "C", "Q"]
    )


# -------------------------------------------------
# Prediction Button
# -------------------------------------------------

if st.button("🔮 Predict Survival", use_container_width=True):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked]
    })


    # Make prediction
    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]


    survival_probability = probability[1] * 100
    non_survival_probability = probability[0] * 100


    # -------------------------------------------------
    # Display Result
    # -------------------------------------------------

    st.divider()

    st.subheader("📊 Prediction Result")


    if prediction == 1:

        st.success(
            "🟢 Prediction: Passenger would likely survive."
        )

    else:

        st.error(
            "🔴 Prediction: Passenger would likely not survive."
        )


    # Probability
    col3, col4 = st.columns(2)


    with col3:

        st.metric(
            "Survival Probability",
            f"{survival_probability:.2f}%"
        )


    with col4:

        st.metric(
            "Non-Survival Probability",
            f"{non_survival_probability:.2f}%"
        )


    # -------------------------------------------------
    # Display Input
    # -------------------------------------------------

    st.subheader("Passenger Details")

    st.dataframe(
        input_data,
        use_container_width=True,
        hide_index=True
    )


# -------------------------------------------------
# Footer
# -------------------------------------------------

st.divider()

st.caption(
    "Titanic Survival Prediction | Machine Learning + Streamlit"
)
