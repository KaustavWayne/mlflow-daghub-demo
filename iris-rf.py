import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
import shutil  # ✅ import shutil
import os

# Initialize DagsHub integration
dagshub.init(repo_owner='KaustavWayne', repo_name='mlflow-daghub-demo', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/KaustavWayne/mlflow-daghub-demo.mlflow")

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

max_depth = 1
n_estimators = 100

mlflow.set_experiment('iris-rf')

with mlflow.start_run() as run:

    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics & params
    mlflow.log_metric('accuracy', accuracy)

    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # Confusion Matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")

    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)

    # ✅ Save model with unique run_id folder
    run_id = run.info.run_id
    model_path = f"random_forest_model_{run_id}"

    mlflow.sklearn.save_model(sk_model=rf, path=model_path)
    mlflow.log_artifact(model_path)

    # Delete the local folder after logging
    shutil.rmtree(model_path)  # ✅ removes folder from VSCode project

    # Tags
    mlflow.set_tag('author', 'priyanka')
    mlflow.set_tag('model', 'random_forest')

    print('accuracy', accuracy)
    print(f"Model logged and local folder '{model_path}' deleted.")
