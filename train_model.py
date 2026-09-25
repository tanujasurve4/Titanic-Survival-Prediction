import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# 1. Load Titanic dataset
df = pd.read_csv("train.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 2. Select features
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


# 3. Separate numerical and categorical features
numeric_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

categorical_features = [
    "Sex",
    "Embarked"
]


# 4. Preprocessing for numerical data
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])


# 5. Preprocessing for categorical data
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# 6. Combine preprocessing
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# 7. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# 8. Create complete pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# 9. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 10. Train the model
print("Training model...")

pipeline.fit(X_train, y_train)

print("Model training completed!")


# 11. Make predictions
y_pred = pipeline.predict(X_test)


# 12. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")


# 13. Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 14. Save trained model
joblib.dump(pipeline, "titanic_model.pkl")

print("\nModel saved successfully!")
print("File created: titanic_model.pkl")
