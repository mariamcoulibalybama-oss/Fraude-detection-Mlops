import mlflow

# 🔥 IMPORTANT : MLflow local
mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.artifacts.download_artifacts(
    artifact_uri="runs:/115e51ff80ce4fb288eb16585ce8bbe6/model",
    dst_path="models"
)

print("✅ Modèle téléchargé dans ./models")