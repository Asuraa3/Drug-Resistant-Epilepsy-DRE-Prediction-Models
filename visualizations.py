import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# ------------------------------------------------------------------------------
# Load the model and data
# ------------------------------------------------------------------------------
output_dir = "output"
features_path = os.path.join(output_dir, "features.csv")
labels_path = os.path.join(output_dir, "labels.npy")
model_path = os.path.join(output_dir, "best_model.joblib")

features = pd.read_csv(features_path)
labels = np.load(labels_path)
model = joblib.load(model_path)

# Load and apply the scaler used during training
scaler_path = os.path.join(output_dir, "scaler.joblib")
scaler = joblib.load(scaler_path)
X = scaler.transform(features)

y = labels

# ------------------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------------------
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

# ------------------------------------------------------------------------------
# Confusion Matrix 
# ------------------------------------------------------------------------------
cm = confusion_matrix(y, y_pred)

# Shape verification
if cm.shape == (2, 2):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-DRE", "DRE"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.show()
else:
    print("Warning: confusion matrix shape not (2,2). Possible missing class in predictions.")
    print(cm)

# ------------------------------------------------------------------------------
# ROC  curve
# ------------------------------------------------------------------------------
if y_prob is not None:
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.show()
else:
    print("Model does not support predict_proba, skipping ROC curve.")

# ------------------------------------------------------------------------------
# Feature Importances
# ------------------------------------------------------------------------------
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    feature_names = features.columns
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"))
    plt.show()
else:
    print("Model does not support feature importances.")

# ------------------------------------------------------------------------------
# Save Predictions
# ------------------------------------------------------------------------------
print("\nExamples of predictions:")
for i in range(min(10, len(y))):
    print(f"Sample {i:2d} | Label = {y[i]} | Prediction = {y_pred[i]}")

print("\n Visualizations saved to the 'output/' folder.")

