import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


os.makedirs("test_outputs", exist_ok=True)


def main():
    # === Paths to pre-extracted features and labels ===
    FEATURES_CSV = 'output/features.csv'
    LABELS_NPY = 'output/labels.npy' 

    # Load features and labels
    feature_df = pd.read_csv(FEATURES_CSV)
    labels = np.load(LABELS_NPY)

    print(f"Loaded features shape: {feature_df.shape}")
    print(f"Loaded labels shape: {labels.shape}")
    print(f"Label distribution: {np.bincount(labels)}")

    # Clean data: replace inf and NaNs
    feature_df.replace([np.inf, -np.inf], 0, inplace=True)
    feature_df.fillna(0, inplace=True)

    X = feature_df.values
    y = labels

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def evaluate_model(model, X, y):
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        return scores.mean(), scores.std()

    # Hyperparameter grids
    tree_bagger_params = {
        'max_depth': [5, 10, 15, 20],
        'n_estimators': [50, 100, 150]
    }

    svm_params = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear']
    }

    knn_params = {
        'n_neighbors': [3, 5, 7, 9]
    }

    logreg_params = {
        'C': [0.01, 0.1, 1, 10]
    }

    print("Testing Tree Bagger hyperparameters...")
    best_score = 0
    best_params = None
    for max_depth in tree_bagger_params['max_depth']:
        for n_estimators in tree_bagger_params['n_estimators']:
            model = BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=42, max_depth=max_depth, class_weight='balanced'),
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1
            )
            mean_score, std_score = evaluate_model(model, X, y)
            print(f"max_depth={max_depth}, n_estimators={n_estimators} -> CV Accuracy: {mean_score:.3f} ± {std_score:.3f}")
            if mean_score > best_score:
                best_score = mean_score
                best_params = {'max_depth': max_depth, 'n_estimators': n_estimators}
    print(f"Best Tree Bagger params: {best_params} with CV accuracy {best_score:.3f}\n")
    #Visualizations for TREE Bagger
    plt.figure()
    for max_depth in tree_bagger_params['max_depth']:
        accuracies = []
        stds = []
        for n_estimators in tree_bagger_params['n_estimators']:
            model = BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=42, max_depth=max_depth, class_weight='balanced'),
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1
            )
            mean_score, std_score = evaluate_model(model, X, y)
            accuracies.append(mean_score)
            stds.append(std_score)
        plt.errorbar(tree_bagger_params['n_estimators'], accuracies, yerr=stds, label=f'Max Depth {max_depth}', capsize=5)
    plt.title('Tree Bagger Accuracy vs N Estimators')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Cross-Validated Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("test_outputs", "tree_bagger_accuracy.png"))
    plt.close()     

    print("Testing SVM hyperparameters...")
    best_score = 0
    best_params = None
    for C in svm_params['C']:
        for kernel in svm_params['kernel']:
            model = SVC(probability=True, random_state=42, kernel=kernel, C=C, class_weight='balanced')
            mean_score, std_score = evaluate_model(model, X, y)
            print(f"C={C}, kernel={kernel} -> CV Accuracy: {mean_score:.3f} ± {std_score:.3f}")
            if mean_score > best_score:
                best_score = mean_score
                best_params = {'C': C, 'kernel': kernel}
    print(f"Best SVM params: {best_params} with CV accuracy {best_score:.3f}\n")
    #Visualizations for SVM
    plt.figure()
    for kernel in svm_params['kernel']:
        accuracies = []
        stds = []
        for C in svm_params['C']:
            model = SVC(probability=True, random_state=42, kernel=kernel, C=C, class_weight='balanced')
            mean_score, std_score = evaluate_model(model, X, y)
            accuracies.append(mean_score)
            stds.append(std_score)
        plt.errorbar(svm_params['C'], accuracies, yerr=stds, label=f'kernel {kernel}', capsize=5)
    plt.xscale("log")
    plt.title('SVM Accuracy vs C')
    plt.xlabel("C")
    plt.ylabel('Cross-Validated Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("test_outputs", "svm_accuracy.png"))
    plt.close()

    print("Testing KNN hyperparameters...")
    best_score = 0
    best_params = None
    for n_neighbors in knn_params['n_neighbors']:
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        mean_score, std_score = evaluate_model(model, X, y)
        print(f"n_neighbors={n_neighbors} -> CV Accuracy: {mean_score:.3f} ± {std_score:.3f}")
        if mean_score > best_score:
            best_score = mean_score
            best_params = {'n_neighbors': n_neighbors}
    print(f"Best KNN params: {best_params} with CV accuracy {best_score:.3f}\n")

    #Visualizations for KNN
    plt.figure()
    accuracies = []
    stds = []
    for k in knn_params['n_neighbors']:
        model = KNeighborsClassifier(n_neighbors=k)
        mean_score, std_score = evaluate_model(model, X, y)
        accuracies.append(mean_score)
        stds.append(std_score)
    plt.errorbar(knn_params['n_neighbors'], accuracies, yerr=stds, capsize=5)
    plt.title('KNN Accuracy')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Cross-Validated Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("test_outputs", "knn_accuracy.png"))
    plt.close()


    print("Testing Logistic Regression hyperparameters...")
    best_score = 0
    best_params = None
    for C in logreg_params['C']:
        model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced', C=C)
        mean_score, std_score = evaluate_model(model, X, y)
        print(f"C={C} -> CV Accuracy: {mean_score:.3f} ± {std_score:.3f}")
        if mean_score > best_score:
            best_score = mean_score
            best_params = {'C': C}
    print(f"Best Logistic Regression params: {best_params} with CV accuracy {best_score:.3f}\n")
    #Visualizations for Logistic Regression
    plt.figure()
    accuracies = []
    stds = []
    for C in logreg_params['C']:
        model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced', C=C)
        mean_score, std_score = evaluate_model(model, X, y)
        accuracies.append(mean_score)
        stds.append(std_score)
    plt.errorbar(logreg_params['C'], accuracies, yerr=stds, capsize=5)
    plt.xscale("log")
    plt.title('Logistic Regression Accuracy')
    plt.xlabel("C")
    plt.ylabel('Cross-Validated Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("test_outputs", "logistic_regression_accuracy.png"))
    plt.close()

if __name__ == "__main__":
    main()
