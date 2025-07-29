"""
MNE-Enhanced Drug-Resistant Epilepsy Predictor
Based on: "Machine learning-based algorithm of drug-resistant prediction
in newly diagnosed patients with temporal lobe epilepsy"

This implementation follows the exact methodology from the research paper:
1. EEG preprocessing (0.1-45 Hz bandpass, frequency decomposition)
2. Phase Lag Index (PLI) calculation for connectivity
3. Graph theory metrics (clustering, path length, efficiency, modularity)
4. Frontotemporal network analysis (key finding)
5. Multiple ML algorithms (Tree Bagger achieved 91.5% accuracy)
6. 5-fold cross-validation
"""
import joblib
import numpy as np
import pandas as pd
import mne

from mne.time_frequency import psd_array_welch
from scipy.signal import hilbert
import os
import glob
from pathlib import Path
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import for BalancedBaggingClassifier and RUSBoostClassifier
from imblearn.ensemble import BalancedBaggingClassifier, RUSBoostClassifier

class PaperMethodologyDREPredictor:
    """
    Drug-Resistant Epilepsy Predictor following  paper methodology
    
    Paper Steps Implemented:
    1. EEG Data Processing (0.1-45 Hz, frequency bands)
    2. Phase Lag Index (PLI) calculation
    3. Network construction (25×25 whole brain, 16×16 frontotemporal)
    4. Graph theory metrics extraction
    5. Machine learning with Tree Bagger (best: 91.5% accuracy)
    """
    
    def __init__(self, sfreq: int = 256):
        self.sfreq = sfreq
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        
        # Paper's exact frequency bands (Hz)
        self.freq_bands = {
            'delta': (1, 4),
            'theta': (4, 7),     # Paper: θ (4-7 Hz)
            'alpha': (8, 13),    # Paper: α (8-13 Hz)
            'beta': (14, 30),    # Paper: β (14-30 Hz)
            'gamma': (30, 45)    # Paper: γ (30-45 Hz)
        }
        
        # Paper's frontotemporal electrodes (IFCN system)
        self.frontotemporal_channels = [
            'Fp1', 'Fp2', 'F3', 'F4', 'F9', 'F10',
            'T9', 'T10', 'P9', 'P10', 'F7', 'F8',
            'T7', 'T8', 'P7', 'P8'
        ]
        
        # Paper's ML algorithms
        self.models = {
            'tree_bagger': BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=42, max_depth=15 , class_weight='balanced'),
                n_estimators=150,
                random_state=42,
                n_jobs=-1
            ),
            'balanced_tree_bagger': BalancedBaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=42, max_depth=15),
                n_estimators=150,
                random_state=42,
                n_jobs=-1,
                sampling_strategy='auto'
            ),
            'naive_bayes': GaussianNB(),
            'svm': SVC(probability=True, random_state=42, kernel='rbf', class_weight='balanced'),
            'knn': KNeighborsClassifier(n_neighbors=9),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            'linear_discriminant': LinearDiscriminantAnalysis(),
            'subspace_knn': BaggingClassifier(
                estimator=KNeighborsClassifier(n_neighbors=3),
                n_estimators=50,
                max_features=0.8,
                random_state=42
            ),
            # NEW: RUSBoosted Tree from the paper
            'rusboost_tree': RUSBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=5, random_state=42), # Using a shallow tree as base estimator
                n_estimators=100,
                random_state=42
            )
        }
    
    def load_edf_with_mne(self, file_path: str, duration_seconds: int = 60) -> mne.io.Raw:
        """
        STEP 1: Load and preprocess EEG data following paper methodology
        
        Paper preprocessing:
        - 0.1-45 Hz band-pass filter
        - Decomposition into frequency bands
        - Common average reference
        """
        print(f"Loading: {os.path.basename(file_path)}")
        
        try:
            # Load EDF with MNE
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            
            print(f"  Original: {raw.info['sfreq']} Hz, {len(raw.ch_names)} channels, {raw.times[-1]:.1f}s")
            
            # Resample to standard frequency
            if raw.info['sfreq'] != self.sfreq:
                raw.resample(self.sfreq)
            
            # Crop to specified duration
            if duration_seconds and raw.times[-1] > duration_seconds:
                raw.crop(tmax=duration_seconds)
            
            # PAPER STEP 1: Apply 0.1-45 Hz band-pass filter
            raw.filter(l_freq=0.1, h_freq=45, verbose=False)
            
            # Remove power line noise
            raw.notch_filter(freqs=[50, 60], verbose=False)
            
            # Set average reference
            raw.set_eeg_reference('average', projection=True, verbose=False)
            raw.apply_proj(verbose=False)
            
            print(f"  Preprocessed: {raw.get_data().shape}")
            
            return raw
            
        except Exception as e:
            print(f"  ERROR: {e}")
            return None
    
    def calculate_phase_lag_index(self, data: np.ndarray) -> np.ndarray:
        """
        STEP 2: Calculate Phase Lag Index (PLI) - Paper's core connectivity measure
        
        Paper formula: PLI = |E{sgn(Δφ)}|
        Where Δφ is the phase difference between signals
        """
        n_channels, n_times = data.shape
        pli_matrix = np.zeros((n_channels, n_channels))
        
        # Get phases using Hilbert transform
        analytic_signals = np.array([hilbert(data[i]) for i in range(n_channels)])
        phases = np.angle(analytic_signals)
        
        # Calculate PLI for each channel pair
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                # Phase difference
                phase_diff = phases[i] - phases[j]
                
                # PLI calculation (exact paper formula)
                pli_value = np.abs(np.mean(np.sign(np.sin(phase_diff))))
                
                pli_matrix[i, j] = pli_value
                pli_matrix[j, i] = pli_value
        
        return pli_matrix
    
    def extract_graph_theory_metrics(self, connectivity_matrix: np.ndarray, prefix: str) -> Dict[str, float]:
        """
        STEP 3: Extract graph theory metrics from paper Table 1
        
        Paper metrics:
        - Clustering Coefficient (CC)
        - Path Length (PL) 
        - Global Efficiency
        - Modularity (Q)
        - Small-worldness (σ)
        """
        features = {}
        
        try:
            # Threshold matrix (paper methodology)
            threshold = np.percentile(connectivity_matrix[connectivity_matrix > 0], 75)
            binary_matrix = (connectivity_matrix > threshold).astype(int)
            np.fill_diagonal(binary_matrix, 0)
            
            # Create graph
            G = nx.from_numpy_array(binary_matrix)
            
            # Paper Metric 1: Clustering Coefficient (CC)
            clustering_coeffs = list(nx.clustering(G).values())
            clustering = np.mean(clustering_coeffs) if clustering_coeffs else 0.0
            
            # Paper Metric 2: Path Length (PL)
            if nx.is_connected(G):
                path_length = nx.average_shortest_path_length(G)
            else:
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                path_length = nx.average_shortest_path_length(subgraph)
            
            # Paper Metric 3: Global Efficiency
            efficiency = nx.global_efficiency(G)
            
            # Paper Metric 4: Modularity (Q)
            communities = nx.community.greedy_modularity_communities(G)
            modularity = nx.community.modularity(G, communities)
            
            # Store metrics with paper naming
            features.update({
                f'{prefix}_clustering_coefficient': clustering,
                f'{prefix}_path_length': path_length,
                f'{prefix}_global_efficiency': efficiency,
                f'{prefix}_modularity': modularity
            })
            
        except Exception as e:
            print(f"    Warning: Graph metrics calculation failed: {e}")
            features.update({
                f'{prefix}_clustering_coefficient': 0.0,
                f'{prefix}_path_length': 0.0,
                f'{prefix}_global_efficiency': 0.0,
                f'{prefix}_modularity': 0.0
            })
        
        return features
    
    def extract_frontotemporal_features(self, raw: mne.io.Raw) -> Dict[str, float]:
        """
        STEP 4: Extract frontotemporal network features (Paper's key finding)
        
        Paper finding: "frontotemporal EEG features significantly enhanced 
        classification performance, particularly in θ-band networks"
        """
        features = {}
        
        # Find available frontotemporal channels
        frontotemporal_electrodes = [
                'FP1', 'FP2', 'F3', 'F4', 'F7', 'F8', 'F9', 'F10',
                'T3', 'T4', 'T7', 'T8', 'T9', 'T10',
                'P7', 'P8', 'P9', 'P10'
        ]

        ft_indices = []
        for i, ch_name in enumerate(raw.ch_names):
            ch_upper = ch_name.upper().replace(' ', '').replace('-', '')
            if ch_upper in frontotemporal_electrodes:
                ft_indices.append(i)
        
        # Fallback: use frontal and temporal channels
        if len(ft_indices) < 4:
            for i, ch_name in enumerate(raw.ch_names):
                if ch_name.upper().startswith(('F', 'T')) and i not in ft_indices:
                    ft_indices.append(i)
                    if len(ft_indices) >= 8:
                        break
        
        if len(ft_indices) >= 2:
            print(f"    Using {len(ft_indices)} frontotemporal channels")
            
            # PAPER KEY FINDING: Focus on theta band (4-7 Hz)
            raw_theta = raw.copy().filter(l_freq=4, h_freq=7, verbose=False)
            ft_data = raw_theta.get_data()[ft_indices, :]
            
            # Calculate PLI for frontotemporal theta network
            ft_pli = self.calculate_phase_lag_index(ft_data)
            
            # Extract PLI statistics
            pli_values = ft_pli[np.triu_indices_from(ft_pli, k=1)]
            
            features.update({
                'ft_theta_pli_mean': np.mean(pli_values),
                'ft_theta_pli_std': np.std(pli_values),
                'ft_theta_pli_max': np.max(pli_values),
                'ft_theta_pli_min': np.min(pli_values),
                'ft_theta_connectivity_strength': np.sum(pli_values),
                'ft_theta_network_density': np.mean(pli_values > 0.1)
            })
            
            # Add graph theory metrics for frontotemporal network
            ft_graph_metrics = self.extract_graph_theory_metrics(ft_pli, 'ft_theta')
            features.update(ft_graph_metrics)
            
            print(f"    Frontotemporal theta PLI: {np.mean(pli_values):.3f}")
        
        return features
    
    def extract_frequency_band_features(self, raw: mne.io.Raw) -> Dict[str, float]:
        """
        STEP 5: Extract features from all frequency bands (Paper Table 1)
        
        Paper bands: δ (1-4), θ (4-7), α (8-13), β (14-30), γ (30-45) Hz
        """
        features = {}
        
        for band_name, (fmin, fmax) in self.freq_bands.items():
            try:
                # Filter to specific frequency band
                band_raw = raw.copy().filter(l_freq=fmin, h_freq=fmax, verbose=False)
                band_data = band_raw.get_data()
                
                # Calculate PLI for this frequency band
                band_pli = self.calculate_phase_lag_index(band_data)
                pli_values = band_pli[np.triu_indices_from(band_pli, k=1)]
                
                # Extract features for this band
                features.update({
                    f'{band_name}_pli_mean': np.mean(pli_values),
                    f'{band_name}_pli_std': np.std(pli_values),
                    f'{band_name}_connectivity_strength': np.sum(pli_values)
                })
                
                # Add graph metrics for this band
                band_graph_metrics = self.extract_graph_theory_metrics(band_pli, band_name)
                features.update(band_graph_metrics)
                
            except Exception as e:
                print(f"    Warning: {band_name} band processing failed: {e}")
    
        return features
    
    def extract_whole_brain_features(self, raw: mne.io.Raw) -> Dict[str, float]:
        """
        STEP 6: Extract whole-brain network features (Paper's 25×25 matrix)
        """
        features = {}
        
        try:
            # Get whole brain data
            whole_brain_data = raw.get_data()
            
            # Calculate whole brain PLI
            whole_brain_pli = self.calculate_phase_lag_index(whole_brain_data)
            pli_values = whole_brain_pli[np.triu_indices_from(whole_brain_pli, k=1)]
            
            features.update({
                'whole_brain_pli_mean': np.mean(pli_values),
                'whole_brain_pli_std': np.std(pli_values),
                'whole_brain_pli_max': np.max(pli_values),
                'whole_brain_connectivity_strength': np.sum(pli_values)
            })
            
            # Add whole brain graph metrics
            wb_graph_metrics = self.extract_graph_theory_metrics(whole_brain_pli, 'whole_brain')
            features.update(wb_graph_metrics)
            
        except Exception as e:
            print(f"    Warning: Whole brain processing failed: {e}")
        
        return features
    
    def extract_power_features(self, raw: mne.io.Raw) -> Dict[str, float]:
        """
        Extract power spectral density features using MNE
        """
        features = {}
        
        try:
            for band_name, (fmin, fmax) in self.freq_bands.items():
                # Calculate PSD using MNE
                psd, freqs = psd_array_welch(raw.get_data(), sfreq=raw.info['sfreq'], fmin=fmin, fmax=fmax, verbose=False)
                
                features.update({
                    f'{band_name}_power_mean': np.mean(psd),
                    f'{band_name}_power_std': np.std(psd)
                })
        
        except Exception as e:
            print(f"    Warning: Power feature extraction failed: {e}")
        
        return features
    
    def extract_all_features(self, raw: mne.io.Raw) -> Dict[str, float]:
        """
        Extract all 216 features mentioned in the paper
        
        Paper feature structure:
        - 108 whole brain features
        - 108 frontotemporal features  
        - Across 5 frequency bands
        - Multiple connectivity and graph metrics
        """
        print(f"  Extracting paper methodology features...")
        features = {}
        
        # STEP 4: Frontotemporal features (Paper's key finding)
        print(f"    Frontotemporal network analysis...")
        ft_features = self.extract_frontotemporal_features(raw)
        features.update(ft_features)
        print(f"    → {len(ft_features)} features extracted from frontotemporal block")
        
        # STEP 5: Frequency band features
        print(f"    Frequency band analysis...")
        freq_features = self.extract_frequency_band_features(raw)
        features.update(freq_features)
        print(f"    → {len(freq_features)} features extracted from frequency band block")
        
        # STEP 6: Whole brain features
        print(f"    Whole brain network analysis...")
        wb_features = self.extract_whole_brain_features(raw)
        features.update(wb_features)
        print(f"    → {len(wb_features)} features extracted from whole brain block")
        
        # Additional power features
        print(f"    Power spectral analysis...")
        power_features = self.extract_power_features(raw)
        features.update(power_features)
        print(f"    → {len(power_features)} features extracted from power spectral block")
        
        print(f"  Extracted {len(features)} features total")
        
        return features
    
    def load_dataset(self, data_directory: str, labels_file: str = None,  duration_seconds: int = 120) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        # Find EDF files
        edf_files = sorted(glob.glob(os.path.join(data_directory, "*.edf")))
        print(f"Found {len(edf_files)} EDF files")

        # Load labels and patient IDs
        labels_dict = {}
        patient_info_df = None
        if labels_file and os.path.exists(labels_file):
            patient_info_df = pd.read_csv(labels_file)
            # Ensure 'filename', 'label', 'patient_id' are present
            if not {'filename', 'label', 'patient_id'}.issubset(patient_info_df.columns):
                raise ValueError("Labels file must contain 'filename', 'label', and 'patient_id' columns for patient-level splitting. Please update your labels.csv.")

            for _, row in patient_info_df.iterrows():
                filename = Path(str(row['filename'])).stem.strip().lower()
                labels_dict[filename] = (int(row['label']), str(row['patient_id'])) # Store (label, patient_id) tuple
            print(f" Loaded {len(labels_dict)} entries (with patient IDs) from {labels_file}")
            print(f"Sample label keys: {list(labels_dict.keys())[:5]}")
        else:
            print(f"Error: No labels file found or missing 'patient_id' column. Cannot perform patient-level splitting.")
            print("Please ensure your labels.csv exists and contains 'filename', 'label', and 'patient_id' columns.")
            return pd.DataFrame(), np.array([]), np.array([]) # Return empty if critical info is missing

        all_processed_data = [] # Store dictionaries with features, label, and patient_id

        for i, file_path in enumerate(edf_files):
            filename_stem = Path(file_path).stem.strip().lower()
            print(f"\nProcessing file {i + 1}/{len(edf_files)}: {filename_stem}.edf")

            raw = self.load_edf_with_mne(file_path, duration_seconds)

            if raw is not None:
                try:
                    features = self.extract_all_features(raw)
                    
                    if filename_stem in labels_dict:
                        label, patient_id = labels_dict[filename_stem]
                    else:
                        print(f"  WARNING: No label/patient_id found for {filename_stem}. Skipping this file.")
                        continue # Skip file if no label/patient_id information
                    
                    # Add label and patient_id to the features dictionary
                    features['label'] = label
                    features['patient_id'] = patient_id
                    all_processed_data.append(features)
                    
                    print(f"  Label: {'DRE' if label == 1 else 'Non-DRE'}, Patient ID: {patient_id}")

                except Exception as e:
                    print(f" ERROR extracting features from {filename_stem}: {e}")
                    continue
            else:
                print(f"  Skipping file due to loading error: {filename_stem}")
                continue

        if len(all_processed_data) == 0:
            print(" No data processed successfully!")
            return pd.DataFrame(), np.array([]), np.array([])

        print(f"\n Processed {len(all_processed_data)} files successfully")
        
        # Convert list of dictionaries to a DataFrame
        final_df = pd.DataFrame(all_processed_data)
        
        # Extract features, labels, patient_ids
        final_labels = final_df['label'].values
        final_patient_ids = final_df['patient_id'].values
        # Features are all other columns
        final_feature_df = final_df.drop(columns=['label', 'patient_id'])

        # Load consensus features and filter the DataFrame
        consensus_features_path = 'correlation_analysis/consensus_features.csv'
        if os.path.exists(consensus_features_path):
            consensus_features_df = pd.read_csv(consensus_features_path)
            consensus_feature_names = consensus_features_df['feature'].tolist()
            
            # Filter final_feature_df to only include consensus features
            # Ensure all consensus features are in the extracted features, fill with 0 if not (shouldn't happen if extraction is consistent)
            missing_consensus_features = [f for f in consensus_feature_names if f not in final_feature_df.columns]
            if missing_consensus_features:
                print(f"  WARNING: Missing consensus features in extracted data: {missing_consensus_features}. Filling with 0.")
                for f in missing_consensus_features:
                    final_feature_df[f] = 0.0
            
            # Select only the consensus features, maintaining their order
            final_feature_df = final_feature_df[consensus_feature_names]
            print(f"  Filtered features to {len(consensus_feature_names)} consensus features.")
        else:
            print(f"  WARNING: Consensus features file not found at {consensus_features_path}. Using all extracted features.")

        print(f" Label distribution: {np.bincount(final_labels)} (DRE count: {np.sum(final_labels)})")
        print(f"Number of unique patients: {len(np.unique(final_patient_ids))}")

        return final_feature_df, final_labels, final_patient_ids


    def train_and_evaluate(self, feature_df: pd.DataFrame, labels: np.ndarray, patient_ids: np.ndarray) -> Dict:
        """
        STEP 7: Train and evaluate ML models (Paper methodology) with patient-level splitting.
        
        Paper validation: 5-fold cross-validation
        Paper best result: Tree Bagger with 91.5% accuracy
        """
        # Clean data: replace inf and NaNs
        feature_df.replace([np.inf, -np.inf], 0, inplace=True)
        feature_df = feature_df.fillna(0)

        # Save the full feature_df and labels for feature selection script
        os.makedirs('output', exist_ok=True) # Ensure output dir exists
        feature_df.to_csv('output/features.csv', index=False)
        np.save('output/labels.npy', labels)

        print("Labels distribution :", np.bincount(labels))

        self.feature_names = feature_df.columns.tolist()
        print(f"\nTraining with {len(self.feature_names)} features")
        
        # Check for single class
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return {'error': 'single_class', 'message': 'Not enough unique classes (DRE/Non-DRE) for training.'}
        
        # Check for enough unique patients for splitting
        unique_patient_ids = np.unique(patient_ids)
        if len(unique_patient_ids) < 2:
            return {'error': 'single_patient', 'message': 'Not enough unique patients for splitting.'}
        if len(unique_patient_ids) < 5:
            print("WARNING: Fewer than 5 unique patients. Patient-level split might be unstable or result in very small sets.")

        # Determine a single label for each patient for stratified splitting of patients
        
        patient_label_map = pd.Series(labels, index=patient_ids).groupby(level=0).apply(lambda x: x.mode()[0])
        unique_patient_labels = patient_label_map.loc[unique_patient_ids].values

        # Perform patient-level train-test split on unique patient IDs
        train_patient_ids, test_patient_ids, _, _ = train_test_split(
            unique_patient_ids, unique_patient_labels,
            test_size=0.2, # Allocate 20% of patients to the test set
            random_state=42,
            stratify=unique_patient_labels # Stratify to keep DRE/Non-DRE patient ratio in train/test
        )

        # Create masks to select data points (recordings) belonging to train/test patients
        train_mask = np.isin(patient_ids, train_patient_ids)
        test_mask = np.isin(patient_ids, test_patient_ids)

        X_train, y_train = feature_df[train_mask], labels[train_mask]
        X_test, y_test = feature_df[test_mask], labels[test_mask]

        from collections import Counter
        print("\nData repartition (patient-level split of recordings):")
        print(f"Train set: {len(X_train)} recordings from {len(train_patient_ids)} patients. Labels: {Counter(y_train)}")
        print(f"Test set: {len(X_test)} recordings from {len(test_patient_ids)} patients. Labels: {Counter(y_test)}")
        
        # Scale features
        # Note: scaler is fit only on X_train to prevent data leakage
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Convert scaled NumPy arrays back to DataFrames with feature names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_names)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_names)

        # Save the scaler, scaled test features, and test labels
        os.makedirs('output', exist_ok=True) # Ensure output dir exists before saving
        joblib.dump(self.scaler, 'output/scaler.joblib')
        X_test_scaled.to_csv('output/X_test_scaled.csv', index=False)
        np.save('output/y_test.npy', y_test)
        
        # Paper validation: 5-fold cross-validation
        # This CV is now done only on the *training data*
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        results = {}
        best_score = 0
        
        print(f"\n{'='*80}")
        print("PAPER METHODOLOGY RESULTS (with Patient-Level Split)")
        print(f"{'='*80}")
        print(f"{'Model':<20} {'CV Acc':<10} {'Test Acc':<10} {'Sens':<8} {'Spec':<8} {'AUC':<8}")
        print("-" * 80)
        
        for model_name, model in self.models.items():
            try:
                # 5-fold cross-validation on training data
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
                
                # Train model on entire training set
                model.fit(X_train_scaled, y_train)
                y_test_pred = model.predict(X_test_scaled)
                
                # Calculate paper metrics on the held-out test set
                test_accuracy = accuracy_score(y_test, y_test_pred)
                cm = confusion_matrix(y_test, y_test_pred)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                else:
                    tn = fp = fn = tp = 0
                    if np.all(y_test == 1):
                        tp = np.sum(y_test_pred == 1)
                        fn = np.sum(y_test_pred == 0)
                        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                        specificity = 0
                    else:
                        tn = np.sum(y_test_pred == 0)
                        fp = np.sum(y_test_pred == 1)
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                        sensitivity = 0
                
                # AUC
                auc = 0.5
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_test_scaled)[:, 1]
                    try:
                        auc = roc_auc_score(y_test, y_prob)
                    except ValueError: # Handle case where only one class is present in y_test
                        auc = 0.5
                
                results[model_name] = {
                    'cv_accuracy_mean': cv_scores.mean(),
                    'cv_accuracy_std': cv_scores.std(),
                    'test_accuracy': test_accuracy,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'auc': auc
                }
                
                # Print results with paper targets
                print(f"{model_name:<20} {cv_scores.mean():.3f}±{cv_scores.std():.3f} " f"{test_accuracy:.3f}     {sensitivity:.3f}   "f"{specificity:.3f}   {auc:.3f}")
                print("conf mat", cm)
                # Track best model
                if cv_scores.mean() > best_score:
                    best_score = cv_scores.mean()
                    self.best_model = model
                    self.best_model_name = model_name
                    
            except Exception as e:
                print(f"{model_name:<20} ERROR: {e}")
        
        # Show paper targets
        print("-" * 80)
        print(f"{'PAPER TARGETS':<20} {'N/A':<10} {'0.915':<10} {'0.970':<8} {'0.810':<8} {'0.920':<8}")
        print(f"Best model: {self.best_model_name} with CV accuracy: {best_score:.3f}")
        
        # Save the best trained model and feature names (features.csv and labels.npy are no longer the *full* dataset)
        joblib.dump(self.best_model, 'output/best_model.joblib')

        with open('output/feature_names.txt', 'w') as f:
            for name in self.feature_names:
                f.write(name + '\n')
            
        return results
    
    def predict_patient(self, file_path: str) -> Dict:
        """
        Predict DRE for a single patient using trained model
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet")
        
        # Load and process file
        raw = self.load_edf_with_mne(file_path)
        if raw is None:
            raise ValueError(f"Could not load {file_path}")
        
        # Extract features
        features = self.extract_all_features(raw)
        
        # Convert to DataFrame and ensure all features present
        feature_df = pd.DataFrame([features])
        for col in self.feature_names:
            if col not in feature_df.columns:
                feature_df[col] = 0.0
        feature_df = feature_df[self.feature_names]
        
        # Scale and predict
        scaler = joblib.load('output/scaler.joblib')
        feature_scaled = scaler.transform(feature_df)


        prediction = self.best_model.predict(feature_scaled)[0]
        
        if hasattr(self.best_model, 'predict_proba'):
            probability = self.best_model.predict_proba(feature_scaled)[0]
            prob_dre = probability[1]
        else:
            prob_dre = float(prediction)
        
        return {
            'prediction': int(prediction),
            'probability_dre': prob_dre,
            'predicted_class': 'DRE' if prediction == 1 else 'Non-DRE',
            'model_used': self.best_model_name,
            'confidence': prob_dre if prediction == 1 else 1 - prob_dre
        }


def main():
    """
    Main function implementing the complete paper methodology
    """
    
    # =========================
    #  PATHS
    # =========================
    DATA_DIRECTORY = r"C:\Users\User\OneDrive\Desktop\Drug-Resistant-Epilepsy-DRE-Prediction-Models\eeg_data\complete_dataset"
    LABELS_FILE = r"C:\Users\User\OneDrive\Desktop\Drug-Resistant-Epilepsy-DRE-Prediction-Models\labels.csv"

    DURATION_SECONDS = 120  # Analysis duration
    # =========================
    
    print("="*80)
    print("DRUG-RESISTANT EPILEPSY PREDICTION")
    print("Following exact methodology from research paper:")
    print("'Machine learning-based algorithm of drug-resistant prediction")
    print("in newly diagnosed patients with temporal lobe epilepsy'")
    print("="*80)
    
    # Check data directory
    if not os.path.exists(DATA_DIRECTORY):
        print(f" ERROR: Directory not found: {DATA_DIRECTORY}")
        print("Please update DATA_DIRECTORY to point to your EDF files")
        return
    
    # Initialize predictor
    predictor = PaperMethodologyDREPredictor()
    
    
    # Load and process dataset
    print(f"\n STEP 1-6: Loading and processing EDF files...")
    try:
        # load_dataset now returns feature_df, labels, and patient_ids
        feature_df, labels, patient_ids = predictor.load_dataset(
            DATA_DIRECTORY, LABELS_FILE, DURATION_SECONDS
        )
    except ValueError as e: # Catch the ValueError if labels.csv is missing patient_id
        print(f" Error loading dataset: {e}")
        return
    except Exception as e:
        print(f" An unexpected error occurred during dataset loading: {e}")
        return
    
    if feature_df.empty or len(labels) == 0:
        print(" No data processed successfully or critical data missing!")
        return
    
    # Train models
    print(f"\n STEP 7: Training ML models (Paper methodology)...")
    # train_and_evaluate now accepts the patient_ids
    results = predictor.train_and_evaluate(feature_df, labels, patient_ids)
    
    if 'error' in results:
        print(f" Training failed: {results.get('message', 'Reason unknown.')}")
        return
    
    
    print(f"\n Paper methodology analysis complete!")
    print(f"Best model: {predictor.best_model_name}")
    print(f"Target: Tree Bagger with 91.5% accuracy (paper result)")

if __name__ == "__main__":
    main()
