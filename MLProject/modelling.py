import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

with mlflow.start_run():
    # 1. Load Data
    df = pd.read_csv('namadataset_preprocessing/dataset_clean.csv')
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train Model
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluasi
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # 5. Log Parameter, Metrik, dan Model ke MLflow
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("accuracy", accuracy)
    
    # Simpan model dengan MLflow
    mlflow.sklearn.log_model(model, "titanic_model")
    
    print(f"Akurasi Model: {accuracy:.4f}")
    print("Model berhasil dieksekusi melalui MLflow Project!")