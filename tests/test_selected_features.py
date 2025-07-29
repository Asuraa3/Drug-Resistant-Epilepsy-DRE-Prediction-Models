import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import os

def test_feature_selections():
    """
    Test different feature selections to see which performs best
    """
    # Load original data
    features_df = pd.read_csv('output/features.csv')
    labels = np.load('output/labels.npy')
    
    # Clean data
    features_df.replace([np.inf, -np.inf], 0, inplace=True)
    features_df.fillna(0, inplace=True)
    
    # Load different feature selections
    feature_selections = {}
    
    selection_files = {
        'correlation_reduced': 'correlation_analysis/correlation_reduced_features.csv',
        'f_test_selected': 'correlation_analysis/f_test_selected_features.csv',
        'mutual_info_selected': 'correlation_analysis/mutual_info_selected_features.csv',
        'rf_importance_selected': 'correlation_analysis/rf_importance_selected_features.csv',
        'consensus_features': 'correlation_analysis/consensus_features.csv'
    }
    
    for name, file_path in selection_files.items():
        if os.path.exists(file_path):
            if name == 'correlation_reduced':
                # This file contains the actual features, not just names
                selected_df = pd.read_csv(file_path)
                feature_selections[name] = selected_df
            else:
                # These files contain feature names
                selected_features = pd.read_csv(file_path)['feature'].tolist()
                feature_selections[name] = features_df[selected_features]
    
    # Add original features for comparison
    feature_selections['original_all'] = features_df
    
    # Test each feature selection
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Use Tree Bagger (best from your original results)
    model = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42, max_depth=15, class_weight='balanced'),
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )
    
    print(f"{'Selection Method':<25} {'# Features':<12} {'CV Accuracy':<15} {'CV Std':<10}")
    print("-" * 70)
    
    for selection_name, X in feature_selections.items():
        try:
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_scaled, labels, cv=cv, scoring='accuracy')
            
            results[selection_name] = {
                'n_features': X.shape[1],
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores
            }
            
            print(f"{selection_name:<25} {X.shape[1]:<12} {cv_scores.mean():.3f}±{cv_scores.std():.3f}    {cv_scores.std():.3f}")
            
        except Exception as e:
            print(f"{selection_name:<25} ERROR: {e}")
    
    # Find best performing selection
    best_selection = max(results.keys(), key=lambda x: results[x]['cv_mean'])
    print(f"\nBest performing selection: {best_selection}")
    print(f"CV Accuracy: {results[best_selection]['cv_mean']:.3f}±{results[best_selection]['cv_std']:.3f}")
    print(f"Number of features: {results[best_selection]['n_features']}")
    
    # Save results
    results_df = pd.DataFrame({
        'selection_method': list(results.keys()),
        'n_features': [results[k]['n_features'] for k in results.keys()],
        'cv_mean': [results[k]['cv_mean'] for k in results.keys()],
        'cv_std': [results[k]['cv_std'] for k in results.keys()]
    }).sort_values('cv_mean', ascending=False)
    
    results_df.to_csv('correlation_analysis/feature_selection_comparison.csv', index=False)
    
    return results

if __name__ == "__main__":
    results = test_feature_selections()
