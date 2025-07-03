import numpy as np

import connectivity_ml_model
from config import edf_folder, excel_path, min_len, target_freq, num_classes, num_epochs
import os


def main():
    l = os.listdir(edf_folder)
    print(len(l))


    # Generate synthetic data (replace with your real data loading)
    eeg_data_list, channel_names_list, labels = connectivity_ml_model.load_data(
        edf_folder, excel_path, resample=False, target_freq=target_freq
    )

    print(f"Loaded {len(eeg_data_list)} EEG recordings")
    print(f"DRE patients: {np.sum(labels)} ({np.mean(labels ) *100:.1f}%)")
    print(f"Non-DRE patients: {len(labels) - np.sum(labels)} ({( 1 -np.mean(labels) ) *100:.1f}%)")
    print(channel_names_list[0])

    # Initialize predictor
    predictor = connectivity_ml_model.DREPredictor(channel_names=channel_names_list[0])  #usa i canali del primo paziente come riferimento

    # Extract features
    print("\n=== Feature Extraction ===")
    feature_matrix = predictor.extract_features_from_data(eeg_data_list, channel_names_list)
    print(f"Feature matrix shape: {feature_matrix.shape}")

    # Train and evaluate models
    print("\n=== Model Training and Evaluation ===")
    results = predictor.train_and_evaluate(feature_matrix, labels)

    # Print detailed results
    print("\n=== Detailed Results ===")
    print(f"{'Model':<20} {'CV Acc':<10} {'Test Acc':<10} {'Sens':<8} {'Spec':<8} {'AUC':<8}")
    print("-" * 70)

    for model_name, metrics in results.items():
        if 'error' not in metrics:
            print(f"{model_name:<20} {metrics['cv_accuracy_mean']:.3f}±{metrics['cv_accuracy_std']:.3f} "
                  f"{metrics['test_accuracy']:.3f}     {metrics['sensitivity']:.3f}   "
                  f"{metrics['specificity']:.3f}   {metrics['auc']:.3f}")

    print(f"\n=== Testing Prediction ===")
    test_eeg = eeg_data_list[0]
    test_channels = channel_names_list[0]

    prediction_result = predictor.predict_dre(test_eeg, test_channels)

    print(f"Patient prediction:")
    print(f"  Predicted class: {'DRE' if prediction_result['prediction'] == 1 else 'Non-DRE'}")
    print(f"  DRE probability: {prediction_result['probability_dre']:.3f}")
    print(f"  Risk level: {prediction_result['risk_level']}")
    print(f"  Model used: {prediction_result['model_used']}")
    print(f"  Actual label: {'DRE' if labels[0] == 1 else 'Non-DRE'}")

    # Plot results
    try:
        predictor.plot_results(results)
    except:
        print("Plotting not available in this environment")

    print(f"\n=== Expected Performance (from paper) ===")
    print(f"Target Accuracy: 91.5%")
    print(f"Target Sensitivity: 97%")
    print(f"Target Specificity: 81%")
    print(f"Target AUC: 0.92")

    print(f"\nImplementation complete! Best model: {predictor.best_model_name}")


if __name__ == "__main__":
    main()
