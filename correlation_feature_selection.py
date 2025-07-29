import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import os

def analyze_feature_correlations():
    """
    Analyze feature correlations and perform feature selection
    """
    # Load your existing features and labels
    features_df = pd.read_csv('features.csv')
    labels = np.load('output/labels.npy')
    
    print(f"Original features shape: {features_df.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Clean data
    features_df.replace([np.inf, -np.inf], 0, inplace=True)
    features_df.fillna(0, inplace=True)
    
    # Create output directory
    os.makedirs('correlation_analysis', exist_ok=True)
    
    # 1. CORRELATION-BASED FEATURE SELECTION
    print("\n=== CORRELATION-BASED FEATURE SELECTION ===")
    
    # Calculate correlation matrix
    corr_matrix = features_df.corr().abs()
    
    # Find highly correlated feature pairs (threshold > 0.95)
    high_corr_threshold = 0.95
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > high_corr_threshold:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    print(f"Found {len(high_corr_pairs)} highly correlated pairs (>{high_corr_threshold})")
    
    # Remove highly correlated features (keep one from each pair)
    features_to_remove = set()
    for pair in high_corr_pairs:
        # Keep the first feature, remove the second
        features_to_remove.add(pair['feature2'])
    
    print(f"Features to remove due to high correlation: {len(features_to_remove)}")
    
    # Create reduced feature set
    features_reduced = features_df.drop(columns=list(features_to_remove))
    print(f"Reduced features shape: {features_reduced.shape}")
    
    # Save correlation analysis results
    high_corr_df = pd.DataFrame(high_corr_pairs)
    high_corr_df.to_csv('correlation_analysis/high_correlation_pairs.csv', index=False)
    
    # 2. STATISTICAL FEATURE SELECTION
    print("\n=== STATISTICAL FEATURE SELECTION ===")
    
    # F-test based selection
    k_best_f = SelectKBest(score_func=f_classif, k=30)  # Select top 30 features
    X_selected_f = k_best_f.fit_transform(features_reduced, labels)
    selected_features_f = features_reduced.columns[k_best_f.get_support()].tolist()
    
    print(f"F-test selected {len(selected_features_f)} features")
    
    # Mutual information based selection
    k_best_mi = SelectKBest(score_func=mutual_info_classif, k=30)
    X_selected_mi = k_best_mi.fit_transform(features_reduced, labels)
    selected_features_mi = features_reduced.columns[k_best_mi.get_support()].tolist()
    
    print(f"Mutual info selected {len(selected_features_mi)} features")
    
    # 3. TREE-BASED FEATURE IMPORTANCE
    print("\n=== TREE-BASED FEATURE IMPORTANCE ===")
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(features_reduced, labels)
    
    # Get feature importances
    feature_importance = pd.DataFrame({
        'feature': features_reduced.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Select top 30 most important features
    top_30_features = feature_importance.head(30)['feature'].tolist()
    
    print(f"Random Forest selected top 30 features")
    
    # 4. SAVE RESULTS
    print("\n=== SAVING RESULTS ===")
    
    # Save different feature selections
    pd.DataFrame({'feature': selected_features_f}).to_csv('correlation_analysis/f_test_selected_features.csv', index=False)
    pd.DataFrame({'feature': selected_features_mi}).to_csv('correlation_analysis/mutual_info_selected_features.csv', index=False)
    pd.DataFrame({'feature': top_30_features}).to_csv('correlation_analysis/rf_importance_selected_features.csv', index=False)
    feature_importance.to_csv('correlation_analysis/all_feature_importances.csv', index=False)
    
    # Save reduced correlation-based features
    features_reduced.to_csv('correlation_analysis/correlation_reduced_features.csv', index=False)
    
    # 5. VISUALIZATIONS
    print("\n=== CREATING VISUALIZATIONS ===")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_20_importance = feature_importance.head(20)
    sns.barplot(data=top_20_importance, x='importance', y='feature')
    plt.title('Top 20 Most Important Features (Random Forest)')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.savefig('correlation_analysis/feature_importance_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    
    # 6. FEATURE OVERLAP ANALYSIS
    print("\n=== FEATURE OVERLAP ANALYSIS ===")
    
    # Find common features across different selection methods
    common_features = set(selected_features_f) & set(selected_features_mi) & set(top_30_features)
    print(f"Features selected by ALL methods: {len(common_features)}")
    print("Common features:", list(common_features))
    
    # Save consensus features
    pd.DataFrame({'feature': list(common_features)}).to_csv('correlation_analysis/consensus_features.csv', index=False)
    
    # 7. VISUALIZE CONSENSUS FEATURES CORRELATION (NEW)
    if common_features:
        print("\n=== CREATING CONSENSUS FEATURES CORRELATION VISUALIZATION ===")
        consensus_df = features_df[list(common_features)] # Use the original features_df to select
        
        plt.figure(figsize=(10, 8)) # Adjust size as needed for 15 features
        sns.heatmap(consensus_df.corr(), annot=True, cmap='coolwarm', center=0, fmt=".2f", linewidths=.5)
        plt.title('Correlation Matrix - Consensus Features (15 Goated Features)')
        plt.xlabel('Features')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig('correlation_analysis/consensus_features_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: correlation_analysis/consensus_features_correlation_matrix.png")
    else:
        print("No consensus features found to visualize correlation matrix.")
    
    return {
        'original_features': features_df.columns.tolist(),
        'correlation_reduced': features_reduced.columns.tolist(),
        'f_test_selected': selected_features_f,
        'mutual_info_selected': selected_features_mi,
        'rf_importance_selected': top_30_features,
        'consensus_features': list(common_features)
    }

if __name__ == "__main__":
    results = analyze_feature_correlations()
    print(f"\n=== SUMMARY ===")
    print(f"Original features: {len(results['original_features'])}")
    print(f"After correlation reduction: {len(results['correlation_reduced'])}")
    print(f"F-test selection: {len(results['f_test_selected'])}")
    print(f"Mutual info selection: {len(results['mutual_info_selected'])}")
    print(f"Random Forest selection: {len(results['rf_importance_selected'])}")
    print(f"Consensus features: {len(results['consensus_features'])}")
