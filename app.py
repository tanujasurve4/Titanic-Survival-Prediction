import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #f7f9fc;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #14213d;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #5c677d;
        margin-bottom: 25px;
    }

    /* Section headings */
    .section-title {
        font-size: 26px;
        font-weight: 650;
        color: #14213d;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    /* Information cards */
    .info-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e6e9ef;
        text-align: center;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.04);
    }

    .card-title {
        color: #6c757d;
        font-size: 14px;
        font-weight: 600;
    }

    .card-value {
        color: #14213d;
        font-size: 28px;
        font-weight: 700;
        margin-top: 5px;
    }

    /* Prediction box */
    .prediction-box {
        padding: 25px;
        border-radius: 14px;
        background-color: white;
        border: 1px solid #e6e9ef;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #7a7f87;
        font-size: 13px;
        margin-top: 40px;
        padding: 20px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv("train.csv")


@st.cache_resource
def load_model():
    return joblib.load("titanic_model.pkl")


df = load_data()
model = load_model()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🚢 Titanic ML")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "📊 Data Analysis",
        "🔮 Survival Prediction",
        "ℹ️ About Model"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "This application uses a Random Forest "
    "machine learning model to predict Titanic "
    "passenger survival."
)


# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    st.markdown(
        '<div class="main-title">🚢 Titanic Survival Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Machine Learning based passenger survival analysis and prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg",
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">📌 Project Overview</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        This project uses historical Titanic passenger data to analyze
        survival patterns and predict whether a passenger would have
        survived based on their personal and travel information.
        """
    )

    st.markdown(
        '<div class="section-title">📊 Dataset Overview</div>',
        unsafe_allow_html=True
    )

    # Dataset metrics
    total_passengers = len(df)
    total_survivors = int(df["Survived"].sum())
    total_non_survivors = total_passengers - total_survivors
    survival_rate = (total_survivors / total_passengers) * 100

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Total Passengers",
            f"{total_passengers:,}"
        )

    with c2:
        st.metric(
            "Survived",
            f"{total_survivors:,}"
        )

    with c3:
        st.metric(
            "Did Not Survive",
            f"{total_non_survivors:,}"
        )

    with c4:
        st.metric(
            "Overall Survival Rate",
            f"{survival_rate:.1f}%"
        )

    st.markdown(
        '<div class="section-title">🧠 Machine Learning Workflow</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    **Titanic Dataset**
    
    ↓
    
    **Data Cleaning & Preprocessing**
    
    ↓
    
    **Feature Selection**
    
    ↓
    
    **Random Forest Classifier**
    
    ↓
    
    **Model Evaluation**
    
    ↓
    
    **Streamlit Prediction Application**
    """)


# =========================================================
# DATA ANALYSIS PAGE
# =========================================================

elif page == "📊 Data Analysis":

    st.markdown(
        '<div class="main-title">📊 Titanic Data Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Explore passenger information and survival patterns'
        '</div>',
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # Dataset Information
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">📋 Dataset Information</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Rows",
            df.shape[0]
        )

    with c2:
        st.metric(
            "Columns",
            df.shape[1]
        )

    with c3:
        missing_values = int(df.isnull().sum().sum())

        st.metric(
            "Missing Values",
            missing_values
        )

    # -----------------------------------------------------
    # Survival Distribution
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">🧍 Survival Distribution</div>',
        unsafe_allow_html=True
    )

    survival_counts = (
        df["Survived"]
        .value_counts()
        .sort_index()
    )

    survival_chart = pd.DataFrame({
        "Status": [
            "Did Not Survive",
            "Survived"
        ],
        "Passengers": [
            int(survival_counts.get(0, 0)),
            int(survival_counts.get(1, 0))
        ]
    })

    st.bar_chart(
        survival_chart.set_index("Status"),
        use_container_width=True
    )

    # -----------------------------------------------------
    # Gender Analysis
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">👨‍👩‍👧 Survival by Gender</div>',
        unsafe_allow_html=True
    )

    gender_analysis = (
        df.groupby(["Sex", "Survived"])
        .size()
        .unstack(fill_value=0)
    )

    gender_analysis = gender_analysis.rename(
        columns={
            0: "Did Not Survive",
            1: "Survived"
        }
    )

    st.bar_chart(
        gender_analysis,
        use_container_width=True
    )

    # -----------------------------------------------------
    # Passenger Class Analysis
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">🎫 Survival by Passenger Class</div>',
        unsafe_allow_html=True
    )

    class_analysis = (
        df.groupby(["Pclass", "Survived"])
        .size()
        .unstack(fill_value=0)
    )

    class_analysis = class_analysis.rename(
        columns={
            0: "Did Not Survive",
            1: "Survived"
        }
    )

    st.bar_chart(
        class_analysis,
        use_container_width=True
    )

    # -----------------------------------------------------
    # Age Analysis
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">🎂 Age Analysis</div>',
        unsafe_allow_html=True
    )

    age_data = df[["Age", "Survived"]].dropna()

    age_survival = (
        age_data.groupby("Survived")["Age"]
        .mean()
        .reset_index()
    )

    age_survival["Status"] = age_survival["Survived"].map({
        0: "Did Not Survive",
        1: "Survived"
    })

    age_survival = age_survival[
        ["Status", "Age"]
    ]

    st.bar_chart(
        age_survival.set_index("Status"),
        use_container_width=True
    )

    # -----------------------------------------------------
    # Fare Analysis
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">💰 Average Fare by Survival</div>',
        unsafe_allow_html=True
    )

    fare_analysis = (
        df.groupby("Survived")["Fare"]
        .mean()
        .reset_index()
    )

    fare_analysis["Status"] = fare_analysis["Survived"].map({
        0: "Did Not Survive",
        1: "Survived"
    })

    fare_analysis = fare_analysis[
        ["Status", "Fare"]
    ]

    st.bar_chart(
        fare_analysis.set_index("Status"),
        use_container_width=True
    )

    # -----------------------------------------------------
    # Raw Dataset
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">📄 Dataset Preview</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# PREDICTION PAGE
# =========================================================

elif page == "🔮 Survival Prediction":

    st.markdown(
        '<div class="main-title">🔮 Survival Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Enter passenger information to generate a prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">👤 Passenger Information</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # Left Column
    # -----------------------------------------------------

    with col1:

        pclass = st.selectbox(
            "🎫 Passenger Class",
            [1, 2, 3],
            help="1 = First Class, 2 = Second Class, 3 = Third Class"
        )

        sex = st.selectbox(
            "👤 Gender",
            ["female", "male"]
        )

        age = st.number_input(
            "🎂 Age",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=1.0
        )

        sibsp = st.number_input(
            "👨‍👩‍👧 Siblings / Spouses",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )

    # -----------------------------------------------------
    # Right Column
    # -----------------------------------------------------

    with col2:

        parch = st.number_input(
            "👪 Parents / Children",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )

        fare = st.number_input(
            "💰 Fare",
            min_value=0.0,
            max_value=600.0,
            value=32.0,
            step=1.0
        )

        embarked = st.selectbox(
            "⚓ Port of Embarkation",
            ["S", "C", "Q"],
            help="S = Southampton, C = Cherbourg, Q = Queenstown"
        )

    st.markdown("")

    predict_button = st.button(
        "🔮 Predict Survival",
        use_container_width=True,
        type="primary"
    )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    if predict_button:

        input_data = pd.DataFrame({
            "Pclass": [pclass],
            "Sex": [sex],
            "Age": [age],
            "SibSp": [sibsp],
            "Parch": [parch],
            "Fare": [fare],
            "Embarked": [embarked]
        })

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        non_survival_probability = probabilities[0] * 100
        survival_probability = probabilities[1] * 100

        st.divider()

        st.markdown(
            '<div class="section-title">📊 Prediction Result</div>',
            unsafe_allow_html=True
        )

        # -------------------------------------------------
        # Result
        # -------------------------------------------------

        if prediction == 1:

            st.success(
                f"🟢 Prediction: Passenger would likely survive."
            )

        else:

            st.error(
                f"🔴 Prediction: Passenger would likely not survive."
            )

        # -------------------------------------------------
        # Probability Metrics
        # -------------------------------------------------

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "🟢 Survival Probability",
                f"{survival_probability:.2f}%"
            )

        with c2:

            st.metric(
                "🔴 Non-Survival Probability",
                f"{non_survival_probability:.2f}%"
            )

        # -------------------------------------------------
        # Probability Chart
        # -------------------------------------------------

        st.markdown(
            '<div class="section-title">📈 Prediction Probability</div>',
            unsafe_allow_html=True
        )

        probability_df = pd.DataFrame({
            "Outcome": [
                "Survived",
                "Did Not Survive"
            ],
            "Probability": [
                survival_probability,
                non_survival_probability
            ]
        })

        st.bar_chart(
            probability_df.set_index("Outcome"),
            use_container_width=True
        )

        # -------------------------------------------------
        # Passenger Details
        # -------------------------------------------------

        st.markdown(
            '<div class="section-title">📋 Entered Passenger Details</div>',
            unsafe_allow_html=True
        )

        display_data = input_data.copy()

        display_data["Pclass"] = display_data["Pclass"].map({
            1: "1st Class",
            2: "2nd Class",
            3: "3rd Class"
        })

        display_data["Embarked"] = display_data["Embarked"].map({
            "S": "Southampton",
            "C": "Cherbourg",
            "Q": "Queenstown"
        })

        display_data.columns = [
            "Passenger Class",
            "Gender",
            "Age",
            "Siblings / Spouses",
            "Parents / Children",
            "Fare",
            "Embarkation"
        ]

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# ABOUT MODEL PAGE
# =========================================================

elif page == "ℹ️ About Model":

    st.markdown(
        '<div class="main-title">ℹ️ About the Machine Learning Model</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Technical information about the Titanic prediction model'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">🤖 Algorithm</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Random Forest Classifier"
    )

    st.markdown(
        '<div class="section-title">📌 Features Used</div>',
        unsafe_allow_html=True
    )

    features_df = pd.DataFrame({
        "Feature": [
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked"
        ],
        "Description": [
            "Passenger class",
            "Passenger gender",
            "Passenger age",
            "Number of siblings or spouses",
            "Number of parents or children",
            "Passenger ticket fare",
            "Port of embarkation"
        ]
    })

    st.dataframe(
        features_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section-title">⚙️ Preprocessing</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        The model pipeline performs preprocessing before making
        predictions.

        • Missing numerical values are handled using the median.

        • Missing categorical values are handled using the most
        frequent value.

        • Categorical variables such as gender and embarkation
        are converted into numerical form using One-Hot Encoding.

        • A Random Forest Classifier is then used for prediction.
        """
    )

    st.markdown(
        '<div class="section-title">🎯 Target Variable</div>',
        unsafe_allow_html=True
    )

    target_df = pd.DataFrame({
        "Value": [0, 1],
        "Meaning": [
            "Did Not Survive",
            "Survived"
        ]
    })

    st.dataframe(
        target_df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        🚢 Titanic Survival Prediction |
        Machine Learning + Python + Streamlit
        <br>
        Developed as a Machine Learning Mini Project
    </div>
    """,
    unsafe_allow_html=True
)
