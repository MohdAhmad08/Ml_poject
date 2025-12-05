import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import pickle


def main():
    data_path = r"C:\Users\mohda\Downloads\heart_disease_uci.csv"
    df = pd.read_csv(data_path)

    print("\nFirst 5 Rows:")
    print(df.head())

    # Replace null values & convert data types properly
    df = df.fillna(df.mode().iloc[0])
    df = df.infer_objects(copy=False)

    # Convert target column (num) to binary:
    # 0 means NO disease
    # >0 means Disease
    df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)

    # Drop unwanted columns
    df.drop(["id", "dataset"], axis=1, inplace=True)

    # Separate X and y
    X = df.drop("num", axis=1)
    y = df["num"]

    # Identify categorical & numeric columns
    cat_cols = X.select_dtypes(include=['object']).columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns

    print("\nCategorical Columns:", list(cat_cols))
    print("Numeric Columns:", list(num_cols))

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first'), cat_cols)
        ]
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # ----------------- Logistic Regression -----------------
    log_clf = Pipeline([
        ('preprocess', preprocessor),
        ('model', LogisticRegression(max_iter=1000))
    ])

    log_clf.fit(X_train, y_train)
    y_pred_lr = log_clf.predict(X_test)

    print("\n===== Logistic Regression =====")
    print("Accuracy :", accuracy_score(y_test, y_pred_lr))
    print("Precision:", precision_score(y_test, y_pred_lr))
    print(classification_report(y_test, y_pred_lr))

    # ----------------- KNN -----------------
    knn = Pipeline([
        ('preprocess', preprocessor),
        ('model', KNeighborsClassifier())
    ])

    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)

    print("\n===== KNN =====")
    print("Accuracy :", accuracy_score(y_test, y_pred_knn))
    print("Precision:", precision_score(y_test, y_pred_knn))
    print(classification_report(y_test, y_pred_knn))

    # ----------------- SVM + Tuning -----------------
    svm_pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', SVC())
    ])

    params = {
        'model__C': [0.1, 1, 10],
        'model__kernel': ['linear', 'rbf'],
        'model__gamma': ['scale', 'auto']
    }

    grid = GridSearchCV(
        svm_pipeline,
        param_grid=params,
        cv=5,
        scoring='precision',
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    print("\nBest Parameters:", grid.best_params_)

    best_model = grid.best_estimator_
    y_pred_svm = best_model.predict(X_test)

    print("\n===== Tuned SVM Model =====")
    print("Accuracy :", accuracy_score(y_test, y_pred_svm))
    print("Precision:", precision_score(y_test, y_pred_svm))
    print(classification_report(y_test, y_pred_svm))

    # Save model
    pickle.dump(best_model, open("best_heart_model.pkl", "wb"))
    print("\n🏆 Model saved as: best_heart_model.pkl")


if __name__ == "__main__":
    main()
