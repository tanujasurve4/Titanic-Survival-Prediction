import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


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

.stApp {
    background-color: #f5f7fb;
}

/* Main title */
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #14213d;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    font-size: 18px;
    color: #667085;
    margin-bottom: 25px;
}

/* Section title */
.section-title {
    font-size: 27px;
    font-weight: 700;
    color: #14213d;
    margin-top: 25px;
    margin-bottom: 15px;
}

/* Small section title */
.small-title {
    font-size: 21px;
    font-weight: 650;
    color: #14213d;
    margin-top: 18px;
    margin-bottom: 10px;
}

/* Chart title */
.chart-title {
    font-size: 18px;
    font-weight: 650;
    color: #14213d;
    text-align: center;
    margin-bottom: 5px;
}

/* Chart card */
.chart-card {
    background-color: white;
    padding: 10px;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.04);
    margin-bottom: 15px;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #e5e7eb;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.04);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
}

/* Info box */
.info-box {
    background-color: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    margin-bottom: 15px;
}

/* Footer */
.footer {
    text-align: center;
    color: #7b8190;
    font-size: 13px;
    padding: 30px;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv("train.csv")


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("titanic_model.pkl")


df = load_data()
model = load_model()


# =========================================================
# MODEL EVALUATION
# =========================================================

@st.cache_data
def evaluate_model():

    features = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked"
    ]

    X = df[features]
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        cm,
        report
    )


(
    accuracy,
    precision,
    recall,
    f1,
    cm,
    report
) = evaluate_model()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Titanic ML Dashboard")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Data Analysis",
        "🔮 Survival Prediction",
        "📊 Model Performance",
        "ℹ️ About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Machine Learning Model: Random Forest"
)

st.sidebar.caption(
    "Titanic Survival Prediction System"
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
        'Machine Learning based passenger survival analysis and prediction.'
        '</div>',
        unsafe_allow_html=True
    )

    # =====================================================
    # TITANIC IMAGE
    # =====================================================

    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg",
        use_container_width=True
    )

    # =====================================================
    # PROJECT OVERVIEW
    # =====================================================

    st.markdown(
        '<div class="section-title">📌 Project Overview</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        This project uses historical Titanic passenger data to analyze
        survival patterns and predict whether a passenger is likely to
        survive based on their personal and travel information.

        The system uses a Random Forest classification model and provides
        an interactive Streamlit interface for data analysis and prediction.
        """
    )

    # =====================================================
    # DATASET METRICS
    # =====================================================

    st.markdown(
        '<div class="section-title">📊 Dataset Overview</div>',
        unsafe_allow_html=True
    )

    total_passengers = len(df)

    survivors = int(
        df["Survived"].sum()
    )

    non_survivors = (
        total_passengers - survivors
    )

    survival_rate = (
        survivors / total_passengers
    ) * 100

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Total Passengers",
            f"{total_passengers:,}"
        )

    with c2:
        st.metric(
            "Survived",
            f"{survivors:,}"
        )

    with c3:
        st.metric(
            "Did Not Survive",
            f"{non_survivors:,}"
        )

    with c4:
        st.metric(
            "Overall Survival Rate",
            f"{survival_rate:.1f}%"
        )

    # =====================================================
    # SURVIVAL OVERVIEW
    # =====================================================

    st.markdown(
        '<div class="section-title">📈 Survival Overview</div>',
        unsafe_allow_html=True
    )

    home_col1, home_col2 = st.columns([1, 1])

    # =====================================================
    # HOME PIE CHART
    # =====================================================

    with home_col1:

        st.markdown(
            '<div class="chart-title">🥧 Overall Survival Distribution</div>',
            unsafe_allow_html=True
        )

        survival_values = [
            non_survivors,
            survivors
        ]

        survival_labels = [
            "Did Not Survive",
            "Survived"
        ]

        fig, ax = plt.subplots(
            figsize=(5, 3.5)
        )

        ax.pie(
            survival_values,
            labels=survival_labels,
            autopct="%1.1f%%",
            startangle=90,
            explode=(0, 0.04)
        )

        ax.set_aspect("equal")

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    # =====================================================
    # MODEL SUMMARY
    # =====================================================

    with home_col2:

        st.markdown(
            '<div class="chart-title">🤖 Model Summary</div>',
            unsafe_allow_html=True
        )

        st.metric(
            "Algorithm",
            "Random Forest"
        )

        st.metric(
            "Model Accuracy",
            f"{accuracy * 100:.2f}%"
        )

        st.metric(
            "Training Dataset",
            "891 Passengers"
        )

    # =====================================================
    # MACHINE LEARNING WORKFLOW
    # =====================================================

    st.markdown(
        '<div class="section-title">🔄 Machine Learning Workflow</div>',
        unsafe_allow_html=True
    )

    workflow_col1, workflow_col2 = st.columns(2)

    with workflow_col1:

        st.markdown("""
        **1. Dataset Collection**

        Titanic passenger dataset is used for model development.

        **2. Data Preprocessing**

        Missing values are handled and categorical features are encoded.

        **3. Feature Selection**

        Pclass, Sex, Age, SibSp, Parch, Fare and Embarked are selected.
        """)

    with workflow_col2:

        st.markdown("""
        **4. Model Training**

        A Random Forest Classifier is trained on the prepared dataset.

        **5. Model Evaluation**

        Accuracy, Precision, Recall, F1 Score and Confusion Matrix are calculated.

        **6. Prediction**

        The Streamlit application accepts passenger information and generates
        a survival prediction.
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
        'Explore passenger characteristics and historical survival patterns.'
        '</div>',
        unsafe_allow_html=True
    )

    # =====================================================
    # DATASET INFORMATION
    # =====================================================

    st.markdown(
        '<div class="section-title">📋 Dataset Information</div>',
        unsafe_allow_html=True
    )

    missing_values = int(
        df.isnull().sum().sum()
    )

    c1, c2, c3, c4 = st.columns(4)

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
        st.metric(
            "Missing Values",
            missing_values
        )

    with c4:
        st.metric(
            "Model Features",
            7
        )

    # =====================================================
    # SURVIVAL + GENDER
    # =====================================================

    chart_col1, chart_col2 = st.columns(2)

    # =====================================================
    # SURVIVAL DISTRIBUTION
    # =====================================================

    with chart_col1:

        st.markdown(
            '<div class="section-title">🥧 Survival Distribution</div>',
            unsafe_allow_html=True
        )

        survival_counts = (
            df["Survived"]
            .value_counts()
            .sort_index()
        )

        fig, ax = plt.subplots(
            figsize=(5, 3.5)
        )

        ax.pie(
            survival_counts.values,
            labels=[
                "Did Not Survive",
                "Survived"
            ],
            autopct="%1.1f%%",
            startangle=90,
            explode=(0, 0.04)
        )

        ax.set_title(
            "Survival Distribution"
        )

        ax.set_aspect("equal")

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    # =====================================================
    # GENDER ANALYSIS
    # =====================================================

    with chart_col2:

        st.markdown(
            '<div class="section-title">👩👨 Survival by Gender</div>',
            unsafe_allow_html=True
        )

        gender_analysis = (
            df.groupby(
                ["Sex", "Survived"]
            )
            .size()
            .unstack(
                fill_value=0
            )
        )

        gender_analysis = gender_analysis.rename(
            columns={
                0: "Did Not Survive",
                1: "Survived"
            }
        )

        st.bar_chart(
            gender_analysis,
            height=300,
            use_container_width=True
        )

    # =====================================================
    # CLASS + AGE GROUP
    # =====================================================

    chart_col1, chart_col2 = st.columns(2)

    # =====================================================
    # PASSENGER CLASS ANALYSIS
    # =====================================================

    with chart_col1:

        st.markdown(
            '<div class="section-title">🎫 Passenger Class Analysis</div>',
            unsafe_allow_html=True
        )

        class_survival = (
            df.groupby(
                "Pclass"
            )["Survived"]
            .mean()
            * 100
        )

        class_survival = (
            class_survival
            .reset_index()
        )

        class_survival.columns = [
            "Passenger Class",
            "Survival Percentage"
        ]

        fig, ax = plt.subplots(
            figsize=(5, 3.5)
        )

        sns.barplot(
            data=class_survival,
            x="Passenger Class",
            y="Survival Percentage",
            ax=ax
        )

        ax.set_title(
            "Survival Percentage by Passenger Class"
        )

        ax.set_ylim(
            0,
            100
        )

        ax.set_xlabel(
            "Passenger Class"
        )

        ax.set_ylabel(
            "Survival (%)"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    # =====================================================
    # AGE GROUP LINE CHART
    # =====================================================

    with chart_col2:

        st.markdown(
            '<div class="section-title">📈 Survival Trend by Age Group</div>',
            unsafe_allow_html=True
        )

        age_data = df[
            ["Age", "Survived"]
        ].dropna().copy()

        age_data["Age Group"] = pd.cut(
            age_data["Age"],
            bins=[
                0,
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                100
            ],
            labels=[
                "0-10",
                "11-20",
                "21-30",
                "31-40",
                "41-50",
                "51-60",
                "61-70",
                "71+"
            ]
        )

        age_survival = (
            age_data.groupby(
                "Age Group",
                observed=False
            )["Survived"]
            .mean()
            * 100
        )

        fig, ax = plt.subplots(
            figsize=(5, 3.5)
        )

        ax.plot(
            age_survival.index.astype(str),
            age_survival.values,
            marker="o",
            linewidth=2
        )

        ax.set_title(
            "Survival Percentage by Age Group"
        )

        ax.set_xlabel(
            "Age Group"
        )

        ax.set_ylabel(
            "Survival (%)"
        )

        ax.set_ylim(
            0,
            100
        )

        ax.tick_params(
            axis="x",
            rotation=45
        )

        ax.grid(
            alpha=0.25
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    # =====================================================
    # AGE VS FARE
    # =====================================================

    st.markdown(
        '<div class="section-title">📍 Age vs Fare Analysis</div>',
        unsafe_allow_html=True
    )

    scatter_data = df[
        ["Age", "Fare", "Survived"]
    ].dropna()

    fig, ax = plt.subplots(
        figsize=(10, 4.5)
    )

    survived_data = scatter_data[
        scatter_data["Survived"] == 1
    ]

    not_survived_data = scatter_data[
        scatter_data["Survived"] == 0
    ]

    ax.scatter(
        not_survived_data["Age"],
        not_survived_data["Fare"],
        alpha=0.45,
        label="Did Not Survive"
    )

    ax.scatter(
        survived_data["Age"],
        survived_data["Fare"],
        alpha=0.45,
        label="Survived"
    )

    ax.set_xlabel(
        "Age"
    )

    ax.set_ylabel(
        "Fare"
    )

    ax.set_title(
        "Passenger Age vs Fare"
    )

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    # =====================================================
    # AVERAGE FARE
    # =====================================================

    st.markdown(
        '<div class="section-title">💰 Average Fare Analysis</div>',
        unsafe_allow_html=True
    )

    fare_col1, fare_col2 = st.columns([1, 1])

    with fare_col1:

        st.markdown(
            '<div class="chart-title">Average Fare by Survival</div>',
            unsafe_allow_html=True
        )

        fare_data = (
            df.groupby(
                "Survived"
            )["Fare"]
            .mean()
            .reset_index()
        )

        fare_data["Status"] = fare_data[
            "Survived"
        ].map({
            0: "Did Not Survive",
            1: "Survived"
        })

        fig, ax = plt.subplots(
            figsize=(6, 3.5)
        )

        sns.barplot(
            data=fare_data,
            x="Status",
            y="Fare",
            ax=ax
        )

        ax.set_xlabel(
            "Survival Status"
        )

        ax.set_ylabel(
            "Average Fare"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    # =====================================================
    # DATA PREVIEW
    # =====================================================

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
# SURVIVAL PREDICTION PAGE
# =========================================================

elif page == "🔮 Survival Prediction":

    st.markdown(
        '<div class="main-title">🔮 Survival Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Enter passenger information to generate a machine learning estimate.'
        '</div>',
        unsafe_allow_html=True
    )

    # =====================================================
    # INPUT SECTION
    # =====================================================

    st.markdown(
        '<div class="section-title">👤 Passenger Information</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        pclass = st.selectbox(
            "Passenger Class",
            [1, 2, 3],
            help="1 = First Class, 2 = Second Class, 3 = Third Class"
        )

        sex = st.selectbox(
            "Gender",
            ["female", "male"]
        )

        age = st.number_input(
            "Age",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=1.0
        )

        sibsp = st.number_input(
            "Siblings / Spouses",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )

    with col2:

        parch = st.number_input(
            "Parents / Children",
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
            ["S", "C", "Q"],
            format_func=lambda x: {
                "S": "Southampton",
                "C": "Cherbourg",
                "Q": "Queenstown"
            }[x]
        )

    st.markdown("")

    predict_button = st.button(
        "🔮 Predict Survival",
        use_container_width=True,
        type="primary"
    )

    # =====================================================
    # PREDICTION
    # =====================================================

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

        prediction = model.predict(
            input_data
        )[0]

        probabilities = model.predict_proba(
            input_data
        )[0]

        non_survival_probability = (
            probabilities[0] * 100
        )

        survival_probability = (
            probabilities[1] * 100
        )

        st.divider()

        # =================================================
        # RESULT
        # =================================================

        st.markdown(
            '<div class="section-title">🎯 Prediction Result</div>',
            unsafe_allow_html=True
        )

        if prediction == 1:

            st.success(
                "Prediction: Passenger is predicted to survive."
            )

        else:

            st.error(
                "Prediction: Passenger is predicted not to survive."
            )

        # =================================================
        # PROBABILITY METRICS
        # =================================================

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Survival Probability",
                f"{survival_probability:.2f}%"
            )

        with c2:

            st.metric(
                "Non-Survival Probability",
                f"{non_survival_probability:.2f}%"
            )

        # =================================================
        # PROBABILITY CHART
        # =================================================

        st.markdown(
            '<div class="section-title">📊 Prediction Probability</div>',
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

        fig, ax = plt.subplots(
            figsize=(7, 3.8)
        )

        sns.barplot(
            data=probability_df,
            x="Outcome",
            y="Probability",
            ax=ax
        )

        ax.set_title(
            "Model Prediction Probability"
        )

        ax.set_ylabel(
            "Probability (%)"
        )

        ax.set_ylim(
            0,
            100
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        # =================================================
        # PASSENGER SUMMARY
        # =================================================

        st.markdown(
            '<div class="section-title">📋 Passenger Summary</div>',
            unsafe_allow_html=True
        )

        summary = pd.DataFrame({
            "Feature": [
                "Passenger Class",
                "Gender",
                "Age",
                "Siblings / Spouses",
                "Parents / Children",
                "Fare",
                "Embarkation"
            ],
            "Value": [
                pclass,
                sex.title(),
                age,
                sibsp,
                parch,
                f"{fare:.2f}",
                {
                    "S": "Southampton",
                    "C": "Cherbourg",
                    "Q": "Queenstown"
                }[embarked]
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        # =================================================
        # DOWNLOAD RESULT
        # =================================================

        result_data = input_data.copy()

        result_data["Prediction"] = (
            "Survived"
            if prediction == 1
            else "Did Not Survive"
        )

        result_data[
            "Survival Probability (%)"
        ] = round(
            survival_probability,
            2
        )

        result_data[
            "Non-Survival Probability (%)"
        ] = round(
            non_survival_probability,
            2
        )

        csv_data = result_data.to_csv(
            index=False
        )

        st.download_button(
            "📥 Download Prediction Result",
            data=csv_data,
            file_name="titanic_prediction.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.info(
            "ℹ️ The prediction is a machine-learning estimate based "
            "on the information entered. It is not a certainty."
        )


# =========================================================
# MODEL PERFORMANCE PAGE
# =========================================================

elif page == "🤖 Model Performance":

    st.markdown(
        '<div class="main-title">🤖 Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Evaluation of the trained Random Forest classification model.'
        '</div>',
        unsafe_allow_html=True
    )

    # =====================================================
    # EVALUATION METRICS
    # =====================================================

    st.markdown(
        '<div class="section-title">🎯 Evaluation Metrics</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with c2:

        st.metric(
            "Precision",
            f"{precision * 100:.2f}%"
        )

    with c3:

        st.metric(
            "Recall",
            f"{recall * 100:.2f}%"
        )

    with c4:

        st.metric(
            "F1 Score",
            f"{f1 * 100:.2f}%"
        )

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    st.markdown(
        '<div class="section-title">🔲 Confusion Matrix</div>',
        unsafe_allow_html=True
    )

    cm_col1, cm_col2 = st.columns([1, 1])

    with cm_col1:

        fig, ax = plt.subplots(
            figsize=(5.5, 4)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[
                "Did Not Survive",
                "Survived"
            ],
            yticklabels=[
                "Did Not Survive",
                "Survived"
            ],
            ax=ax
        )

        ax.set_xlabel(
            "Predicted"
        )

        ax.set_ylabel(
            "Actual"
        )

        ax.set_title(
            "Confusion Matrix"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    with cm_col2:

        st.markdown(
            '<div class="chart-title">📌 What the Matrix Shows</div>',
            unsafe_allow_html=True
        )

        st.write(
            """
            The confusion matrix compares the actual passenger outcomes
            with the outcomes predicted by the machine learning model.

            **True Negative:** Correctly predicted non-survivor.

            **True Positive:** Correctly predicted survivor.

            **False Positive:** Predicted survivor but passenger did not survive.

            **False Negative:** Predicted non-survivor but passenger survived.
            """
        )

    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    st.markdown(
        '<div class="section-title">📋 Classification Report</div>',
        unsafe_allow_html=True
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    report_df = report_df.round(
        3
    )

    st.dataframe(
        report_df,
        use_container_width=True
    )

    # =====================================================
    # MODEL INFORMATION
    # =====================================================

    st.markdown(
        '<div class="section-title">🌳 Random Forest Information</div>',
        unsafe_allow_html=True
    )

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:

        st.info(
            "**Algorithm**\n\nRandom Forest Classifier"
        )

    with info_col2:

        st.info(
            "**Number of Trees**\n\n200"
        )

    with info_col3:

        st.info(
            "**Random State**\n\n42"
        )

    # =====================================================
    # EVALUATION METHOD
    # =====================================================

    st.markdown(
        '<div class="section-title">📌 Evaluation Method</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        The dataset was divided into training and testing data using an
        80:20 split. Stratified splitting was used so that the proportion
        of survived and non-survived passengers is maintained in both
        datasets.

        The evaluation metrics are calculated on the held-out test data.
        """
    )


# =========================================================
# ABOUT PROJECT PAGE
# =========================================================

elif page == "ℹ️ About Project":

    st.markdown(
        '<div class="main-title">ℹ️ About the Project</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Technical details of the Titanic Survival Prediction system.'
        '</div>',
        unsafe_allow_html=True
    )

    # =====================================================
    # TECHNOLOGY STACK
    # =====================================================

    st.markdown(
        '<div class="section-title">💻 Technology Stack</div>',
        unsafe_allow_html=True
    )

    tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)

    with tech_col1:

        st.info(
            "🐍 **Python**\n\nProgramming Language"
        )

    with tech_col2:

        st.info(
            "🤖 **Scikit-learn**\n\nMachine Learning"
        )

    with tech_col3:

        st.info(
            "📊 **Pandas**\n\nData Analysis"
        )

    with tech_col4:

        st.info(
            "🌐 **Streamlit**\n\nWeb Application"
        )

    # =====================================================
    # FEATURES
    # =====================================================

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

    # =====================================================
    # PREPROCESSING
    # =====================================================

    st.markdown(
        '<div class="section-title">⚙️ Data Preprocessing</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    ### Numerical Features

    Missing numerical values are handled using **median imputation**.

    ### Categorical Features

    Missing categorical values are replaced using the **most frequent value**.

    ### Encoding

    Categorical features such as **Sex** and **Embarked** are converted
    into numerical values using **One-Hot Encoding**.

    ### Classification

    A **Random Forest Classifier** is used to predict the target variable.
    """)

    # =====================================================
    # TARGET VARIABLE
    # =====================================================

    st.markdown(
        '<div class="section-title">🎯 Target Variable</div>',
        unsafe_allow_html=True
    )

    target_df = pd.DataFrame({
        "Value": [
            0,
            1
        ],
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

    # =====================================================
    # PROJECT OBJECTIVE
    # =====================================================

    st.markdown(
        '<div class="section-title">🎯 Project Objective</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        The main objective of this project is to develop a machine learning
        system that can analyze historical Titanic passenger information
        and estimate the probability of passenger survival.

        The project demonstrates the complete machine learning workflow:
        data preprocessing, feature selection, model training, evaluation
        and deployment using Streamlit.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        🚢 Titanic Survival Prediction
        <br>
        Machine Learning • Python • Scikit-learn • Streamlit
        <br><br>
        Developed as a Machine Learning Mini Project
    </div>
    """,
    unsafe_allow_html=True
)
