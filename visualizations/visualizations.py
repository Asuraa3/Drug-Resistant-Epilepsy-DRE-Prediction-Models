import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

def create_visualizations():
    """
    Loads the trained model, scaler, and held-out test data,
    then generates and saves classification-specific visualizations and metrics.
    """
    output_dir = "output" 

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------------------
    # Load the model, scaler, and test data
    # ------------------------------------------------------------------------------
    model_path = os.path.join(output_dir, "best_model.joblib")
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    X_test_path = os.path.join(output_dir, "X_test_scaled.csv")
    y_test_path = os.path.join(output_dir, "y_test.npy")
    feature_names_path = os.path.join(output_dir, "feature_names.txt")

    if not all(os.path.exists(p) for p in [model_path, scaler_path, X_test_path, y_test_path, feature_names_path]):
        print(f"Error: Required files not found in '{output_dir}/'.")
        print("Please ensure 'mne_dre_predictor.py' has been run successfully to generate these files.")
        return

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) # Scaler is loaded but not directly used here as X_test is already scaled
    X_test = pd.read_csv(X_test_path)
    y_test = np.load(y_test_path)

    # Load feature names
    with open(feature_names_path, 'r') as f:
        feature_names = [line.strip() for line in f]

    # NEW: Check for and drop an unnamed index column if present
    if 'Unnamed: 0' in X_test.columns:
        print("DEBUG (visualizations): 'Unnamed: 0' column found. Dropping it.")
        X_test = X_test.drop(columns=['Unnamed: 0'])
        print(f"DEBUG (visualizations): X_test shape after dropping 'Unnamed: 0': {X_test.shape}")

    # Ensure columns match original feature names
    # This line will now work correctly if the number of columns matches
    X_test.columns = feature_names 

    print(f"Loaded test features shape: {X_test.shape}")
    print(f"Loaded test labels shape: {y_test.shape}")
    print(f"Test Label distribution: {np.bincount(y_test)}")

    # ------------------------------------------------------------------------------
    # Prediction on the held-out test set
    # ------------------------------------------------------------------------------
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        print("Warning: Model does not support predict_proba. ROC curve will not be generated.")

    # ------------------------------------------------------------------------------
    # Calculate and Print Classification Metrics
    # ------------------------------------------------------------------------------
    test_accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = 0, 0, 0, 0
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0 # Also known as Recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        print("Warning: Confusion matrix shape not (2,2). Cannot calculate sensitivity/specificity directly.")
        print(cm)
        # Attempt to calculate based on available classes if only one class is predicted/present
        if np.all(y_test == 1): # All actual are positive
            tp = np.sum(y_pred == 1)
            fn = np.sum(y_pred == 0)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = 0 # Cannot calculate if no true negatives
        elif np.all(y_test == 0): # All actual are negative
            tn = np.sum(y_pred == 0)
            fp = np.sum(y_pred == 1)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = 0 # Cannot calculate if no true positives
        else: # Mixed classes, but prediction might be single class
            # Fallback to sklearn's recall/precision if cm.ravel() fails
            sensitivity = recall_score(y_test, y_pred, pos_label=1, average='binary', zero_division=0)
            specificity = recall_score(y_test, y_pred, pos_label=0, average='binary', zero_division=0)


    auc_score = 0.5
    if y_prob is not None:
        try:
            auc_score = roc_auc_score(y_test, y_prob)
        except ValueError as e:
            print(f"Warning: Could not calculate AUC. {e}")
            auc_score = 0.5 # Default to chance if calculation fails

    metrics_string = (
        f"Model Performance on Held-Out Test Set:\n"
        f"----------------------------------------\n"
        f"Accuracy: {test_accuracy:.3f}\n"
        f"Sensitivity (Recall): {sensitivity:.3f}\n"
        f"Specificity: {specificity:.3f}\n"
        f"AUC: {auc_score:.3f}\n"
        f"----------------------------------------\n"
        f"Confusion Matrix Details:\n"
        f"  True Positives (TP): {tp}\n"
        f"  False Negatives (FN): {fn}\n"
        f"  False Positives (FP): {fp}\n"
        f"  True Negatives (TN): {tn}\n"
    )
    print(metrics_string)

    with open(os.path.join(output_dir, "model_classification_metrics.txt"), "w") as f:
        f.write(metrics_string)

    # ------------------------------------------------------------------------------
    # Confusion Matrix Plot
    # ------------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-DRE", "DRE"])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'confusion_matrix.png')}")

    # ------------------------------------------------------------------------------
    # ROC Curve Plot
    # ------------------------------------------------------------------------------
    if y_prob is not None:
        plt.figure(figsize=(8, 6))
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.2f})")
        plt.plot([0, 1], [0, 1], "k--", label="Chance")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "roc_curve.png"))
        plt.close()
        print(f"Saved: {os.path.join(output_dir, 'roc_curve.png')}")
    else:
        print("Skipping ROC curve: Model does not support probability predictions.")

    # ------------------------------------------------------------------------------
    # Feature Importances Plot
    # ------------------------------------------------------------------------------
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 7)) # Increased figure size for better readability
        sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette='viridis')
        plt.title("Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_importances.png"))
        plt.close()
        print(f"Saved: {os.path.join(output_dir, 'feature_importances.png')}")
    else:
        print("Skipping Feature Importances: Model does not support feature_importances_ attribute.")

    print(f"\nAll classification visualizations and metrics saved to the '{output_dir}/' folder.")

if __name__ == '__main__':
    # This block ensures create_visualizations() is called when the script is run directly
    create_visualizations()
