"""
Connectivity-Based Machine Learning Model for DRE Prediction
Based on: "Machine learning-based algorithm of drug-resistant prediction 
in newly diagnosed patients with temporal lobe epilepsy"

This implementation focuses on Phase Lag Index (PLI) and graph theory features
from frontotemporal networks in the theta band (4-8 Hz).
"""
import numpy as np
import pandas as pd
import mne
from mne.connectivity import phase_lag_index, spectral_connectivity_epochs
from mne.time_frequency import psd_welch
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
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class EEGConnectivityFeatures:
    """
    Extract EEG connectivity features as described in the paper.
    Focus on Phase Lag Index (PLI) and graph theory metrics.
    """
    
    def __init__(self, sfreq: int = 256):
        self.sfreq = sfreq
        # Standard 10-20 electrode positions
        self.all_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
                            'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 
                            'Fz', 'Cz', 'Pz']
        
        # Frontotemporal channels (key finding from paper)
        self.frontotemporal_channels = ['F7', 'F8', 'T3', 'T4', 'F3', 'F4']
        
    def bandpass_filter(self, data: np.ndarray, low_freq: float, high_freq: float) -> np.ndarray:
        """Apply bandpass filter to EEG data"""
        nyquist = self.sfreq / 2 # Maximum frequency we can detect
        low = low_freq / nyquist # Normalize frenquencies 
        high = high_freq / nyquist

        #Create and apply the filter
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_data = signal.filtfilt(b, a, data, axis=-1)
        return filtered_data
    
    def calculate_phase_lag_index(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate Phase Lag Index (PLI) between all channel pairs.
        PLI measures the asymmetry of the phase difference distribution.
        
        Args:
            data: EEG data (channels x time_points)
            
        Returns:
            PLI matrix (channels x channels)
        """

        n_channels, n_times = data.shape
        pli_matrix = np.zeros((n_channels, n_channels))
        
        # Get the "phase" of each brain wave
        analytic_signals = signal.hilbert(data, axis=-1)
        phases = np.angle(analytic_signals)

        # Compare every pair of electrodes
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                # Phase difference , how different are their phases?
                phase_diff = phases[i] - phases[j]
                
                # PLI calculation - measure consistency of phase difference
                pli_value = np.abs(np.mean(np.sign(np.sin(phase_diff))))
                pli_matrix[i, j] = pli_value
                pli_matrix[j, i] = pli_value  # Symmetric matrix - Mirror the matrix
                
        return pli_matrix
    
    def calculate_clustering_coefficient(self, connectivity_matrix: np.ndarray, threshold_percentile: float = 75) -> float:
        """Calculate average clustering coefficient from connectivity matrix"""
        # Threshold connectivity matrix to create binary graph
        threshold = np.percentile(connectivity_matrix[connectivity_matrix > 0], threshold_percentile)
        binary_matrix = (connectivity_matrix > threshold).astype(int)
        
        # Remove self-connections
        np.fill_diagonal(binary_matrix, 0)
        
        # Create graph and calculate clustering coefficient
        G = nx.from_numpy_array(binary_matrix)
        clustering_coeffs = list(nx.clustering(G).values())
        
        return np.mean(clustering_coeffs) if clustering_coeffs else 0.0
    
    def calculate_shortest_path_length(self, connectivity_matrix: np.ndarray) -> float:
        """Calculate average shortest path length"""
        # Create weighted graph (invert weights since higher connectivity = shorter path)
        weights = 1 / (connectivity_matrix + 1e-10)  # Add small value to avoid division by zero
        np.fill_diagonal(weights, 0)
        
        G = nx.from_numpy_array(weights)
        
        try:
            # Calculate average shortest path length
            if nx.is_connected(G):
                avg_path_length = nx.average_shortest_path_length(G, weight='weight')
            else:
                # For disconnected graphs, calculate for largest component
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                avg_path_length = nx.average_shortest_path_length(subgraph, weight='weight')
        except:
            avg_path_length = np.inf
            
        return avg_path_length
    
    def calculate_global_efficiency(self, connectivity_matrix: np.ndarray) -> float:
        """Calculate global efficiency of the network"""
        G = nx.from_numpy_array(connectivity_matrix)
        return nx.global_efficiency(G)
    
    def calculate_modularity(self, connectivity_matrix: np.ndarray) -> float:
        """Calculate modularity of the network"""
        # Threshold matrix for modularity calculation
        threshold = np.percentile(connectivity_matrix[connectivity_matrix > 0], 75)
        binary_matrix = (connectivity_matrix > threshold).astype(int)
        np.fill_diagonal(binary_matrix, 0)
        
        G = nx.from_numpy_array(binary_matrix)
        
        try:
            communities = nx.community.greedy_modularity_communities(G)
            modularity = nx.community.modularity(G, communities)
        except:
            modularity = 0.0
            
        return modularity
    
    def extract_frontotemporal_features(self, eeg_data: np.ndarray, channel_names: List[str]) -> Dict[str, float]:
        """
        Extract features specifically from frontotemporal network.
        This was the key finding in the paper.
        
        Args:
            eeg_data: EEG data (channels x time_points)
            channel_names: List of channel names
            
        Returns:
            Dictionary of frontotemporal features
        """
        # Find available frontotemporal channels
        available_ft_channels = []
        ft_indices = []
        
        for i, ch_name in enumerate(channel_names):
            if any(ft_ch in ch_name.upper() for ft_ch in ['F7', 'F8', 'T3', 'T4', 'F3', 'F4', 'FT7', 'FT8']):
                available_ft_channels.append(ch_name)
                ft_indices.append(i)
        
        if len(ft_indices) < 2:
            # If no specific frontotemporal channels, use frontal and temporal
            for i, ch_name in enumerate(channel_names):
                if ch_name.upper().startswith(('F', 'T')) and len(ft_indices) < 6:
                    if i not in ft_indices:
                        available_ft_channels.append(ch_name)
                        ft_indices.append(i)
        
        if len(ft_indices) < 2:
            # Fallback: use first few channels
            ft_indices = list(range(min(6, eeg_data.shape[0])))
        
        # Extract frontotemporal data
        ft_data = eeg_data[ft_indices, :]
        
        # Filter to theta band (4-8 Hz) - key finding from paper
        theta_data = self.bandpass_filter(ft_data, 4, 8)
        
        # Calculate PLI for frontotemporal network
        ft_pli = self.calculate_phase_lag_index(theta_data)
        
        # Calculate network metrics
        clustering_coeff = self.calculate_clustering_coefficient(ft_pli)
        path_length = self.calculate_shortest_path_length(ft_pli)
        global_efficiency = self.calculate_global_efficiency(ft_pli)
        modularity = self.calculate_modularity(ft_pli)
        
        # Extract statistical features from PLI matrix
        pli_upper_triangle = ft_pli[np.triu_indices_from(ft_pli, k=1)]
        
        features = {
            'ft_pli_mean': np.mean(pli_upper_triangle), # Average connectivity
            'ft_pli_std': np.std(pli_upper_triangle),   # Variability 
            'ft_pli_max': np.max(pli_upper_triangle),   # Strongest connection
            'ft_pli_min': np.min(pli_upper_triangle),
            'ft_pli_median': np.median(pli_upper_triangle),
            'ft_clustering_coeff': clustering_coeff,
            'ft_path_length': path_length if path_length != np.inf else 0,
            'ft_global_efficiency': global_efficiency,
            'ft_modularity': modularity,
            'ft_connectivity_strength': np.sum(pli_upper_triangle),
            'ft_connectivity_density': np.mean(pli_upper_triangle > 0.1),  # Proportion of strong connections
        }
        
        return features
    
    def extract_frequency_band_features(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """Extract features from different frequency bands"""
        frequency_bands = {
            'delta': (1, 4),    # Deep sleep waves
            'theta': (4, 8),    # Most important according to paper
            'alpha': (8, 13),   # Relaxed awareness
            'beta': (13, 30),   # Active thinking
            'gamma': (30, 45)   # High-level processing
        }
        
        features = {}
        
        for band_name, (fmin, fmax) in frequency_bands.items():
            # Filter to specefic frequency band
            band_data = self.bandpass_filter(eeg_data, fmin, fmax)
            
            # Calculate PLI (connectivity in this frequency)
            band_pli = self.calculate_phase_lag_index(band_data)
            pli_upper = band_pli[np.triu_indices_from(band_pli, k=1)]
            
            # Extract features for this frequency band
            features.update({
                f'{band_name}_pli_mean': np.mean(pli_upper),
                f'{band_name}_pli_std': np.std(pli_upper),
                f'{band_name}_clustering': self.calculate_clustering_coefficient(band_pli),
                f'{band_name}_efficiency': self.calculate_global_efficiency(band_pli)
            })
        
        return features
    
    def extract_all_features(self, eeg_data: np.ndarray, channel_names: List[str]) -> Dict[str, float]:
        """
        Extract all 216 features as mentioned in the paper.
        
        Args:
            eeg_data: EEG data (channels x time_points)
            channel_names: List of channel names
            
        Returns:
            Dictionary of all extracted features
        """
        features = {}
        
        # 1. Frontotemporal features (most important according to paper)
        ft_features = self.extract_frontotemporal_features(eeg_data, channel_names)
        features.update(ft_features)
        
        # 2. Whole brain PLI features
        whole_brain_pli = self.calculate_phase_lag_index(eeg_data)
        pli_upper = whole_brain_pli[np.triu_indices_from(whole_brain_pli, k=1)]
        
        features.update({
            'whole_brain_pli_mean': np.mean(pli_upper),
            'whole_brain_pli_std': np.std(pli_upper),
            'whole_brain_pli_max': np.max(pli_upper),
            'whole_brain_pli_min': np.min(pli_upper),
            'whole_brain_clustering': self.calculate_clustering_coefficient(whole_brain_pli),
            'whole_brain_path_length': self.calculate_shortest_path_length(whole_brain_pli),
            'whole_brain_efficiency': self.calculate_global_efficiency(whole_brain_pli),
            'whole_brain_modularity': self.calculate_modularity(whole_brain_pli)
        })
        
        # 3. Frequency band specific features
        freq_features = self.extract_frequency_band_features(eeg_data)
        features.update(freq_features)
        
        # 4. Additional statistical features
        for i, ch_name in enumerate(channel_names[:min(10, len(channel_names))]):  # Limit to first 10 channels
            ch_data = eeg_data[i, :]
            features.update({
                f'{ch_name}_power': np.var(ch_data),
                f'{ch_name}_skewness': self._calculate_skewness(ch_data),
                f'{ch_name}_kurtosis': self._calculate_kurtosis(ch_data)
            })
        
        return features
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of the data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of the data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3

class DREPredictor:
    """Drug-Resistant Epilepsy Predictor based on connectivity features."""
    
    def __init__(self):
        self.feature_extractor = EEGConnectivityFeatures()
        self.scaler = StandardScaler()
        self.models = self._initialize_models()
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        
    def _initialize_models(self) -> Dict:
        """Initialize all models used in the paper"""
        models = {
            'tree_bagger': BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=42, max_depth=10),
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            'naive_bayes': GaussianNB(),
            'svm': SVC(probability=True, random_state=42, kernel='rbf'),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'linear_discriminant': LinearDiscriminantAnalysis(),
            'subspace_knn': BaggingClassifier(
                estimator=KNeighborsClassifier(n_neighbors=3),
                n_estimators=50,
                max_features=0.8,
                random_state=42,
                n_jobs=-1
            )
        }
        return models
    
    def create_labels_template(self, data_directory: str):
        """Create a CSV template for labeling EDF files as DRE or Non-DRE."""
        edf_files = glob.glob(os.path.join(data_directory, "*.edf"))
        edf_files.extend(glob.glob(os.path.join(data_directory, "*.EDF")))
        
        if len(edf_files) == 0:
            print(f"No EDF files found in {data_directory}")
            return None
        
        filenames = [Path(f).stem for f in sorted(edf_files)]
        labels_df = pd.DataFrame({
            'filename': filenames,
            'label': [0] * len(filenames),
            'notes': [''] * len(filenames)
        })
        
        output_path = os.path.join(data_directory, "labels.csv")
        labels_df.to_csv(output_path, index=False)
        
        print(f"Created labels template: {output_path}")
        print("Please edit this file and set:")
        print("  label = 0 for Non-DRE (drug responsive) patients")
        print("  label = 1 for DRE (drug resistant) patients")
        
        return output_path
    
    def load_edf_files(self, data_directory: str, labels_file: str = None, 
                      duration_seconds: int = 60) -> Tuple[List[np.ndarray], List[List[str]], np.ndarray]:
        """Load EDF files from directory"""
        
        # Check if MNE is available
        try:
            import mne
        except ImportError:
            raise ImportError("MNE package required for EDF loading. Install with: pip install mne")
        
        # Find EDF files
        edf_files = glob.glob(os.path.join(data_directory, "*.edf"))
        edf_files.extend(glob.glob(os.path.join(data_directory, "*.EDF")))
        edf_files = sorted(edf_files)
        
        print(f"Found {len(edf_files)} EDF files in {data_directory}")
        
        if len(edf_files) == 0:
            raise ValueError(f"No EDF files found in {data_directory}")
        
        # Load labels if provided
        labels_dict = {}
        if labels_file and os.path.exists(labels_file):
            labels_df = pd.read_csv(labels_file)
            print(f"Loaded labels from {labels_file}")
            
            filename_col = None
            label_col = None
            
            for col in labels_df.columns:
                if 'file' in col.lower() or 'name' in col.lower():
                    filename_col = col
                if 'label' in col.lower() or 'dre' in col.lower() or 'class' in col.lower():
                    label_col = col
            
            if filename_col and label_col:
                for _, row in labels_df.iterrows():
                    filename = str(row[filename_col]).replace('.edf', '').replace('.EDF', '')
                    labels_dict[filename] = int(row[label_col])
                print(f"Loaded {len(labels_dict)} labels")
        
        # Load EDF files
        eeg_data_list = []
        channel_names_list = []
        labels = []
        
        for i, file_path in enumerate(edf_files):
            try:
                print(f"Loading {i+1}/{len(edf_files)}: {os.path.basename(file_path)}")
                
                raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
                
                if raw.info['sfreq'] != 256:
                    print(f"  Resampling from {raw.info['sfreq']} Hz to 256 Hz")
                    raw.resample(256)
                
                if duration_seconds and raw.times[-1] > duration_seconds:
                    raw.crop(tmax=duration_seconds)
                    print(f"  Cropped to {duration_seconds} seconds")
                
                data = raw.get_data()
                channel_names = [ch.strip().replace(' ', '') for ch in raw.ch_names]
                
                eeg_data_list.append(data)
                channel_names_list.append(channel_names)
                
                filename = Path(file_path).stem
                if filename in labels_dict:
                    label = labels_dict[filename]
                elif any(keyword in filename.lower() for keyword in ['dre', 'resistant']):
                    label = 1
                elif any(keyword in filename.lower() for keyword in ['responsive', 'nondre']):
                    label = 0
                else:
                    label = i % 2
                    print(f"  No label found, using default: {label}")
                
                labels.append(label)
                print(f"  Shape: {data.shape}, Label: {'DRE' if label == 1 else 'Non-DRE'}")
                
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
                continue
        
        print(f"\nSuccessfully loaded {len(eeg_data_list)} files")
        print(f"DRE patients: {np.sum(labels)} ({np.mean(labels)*100:.1f}%)")
        
        return eeg_data_list, channel_names_list, np.array(labels)
    
    def extract_features_from_data(self, eeg_data_list: List[np.ndarray], channel_names_list: List[List[str]]) -> pd.DataFrame:
        """Extract features from list of EEG data arrays."""
        all_features = []
        
        print(f"Extracting features from {len(eeg_data_list)} EEG recordings...")
        
        for i, (eeg_data, channel_names) in enumerate(zip(eeg_data_list, channel_names_list)):
            print(f"Processing recording {i+1}/{len(eeg_data_list)}")
            
            try:
                features = self.feature_extractor.extract_all_features(eeg_data, channel_names)
                all_features.append(features)
            except Exception as e:
                print(f"Error processing recording {i+1}: {e}")
                if all_features:
                    empty_features = {key: 0.0 for key in all_features[0].keys()}
                    all_features.append(empty_features)
                else:
                    continue
        
        feature_df = pd.DataFrame(all_features)
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.fillna(0)
        
        self.feature_names = feature_df.columns.tolist()
        print(f"Extracted {len(self.feature_names)} features")
        
        return feature_df
    
    def train_and_evaluate(self, X: pd.DataFrame, y: np.ndarray, cv_folds: int = 5, test_size: float = 0.2) -> Dict:
        """Train and evaluate all models using cross-validation."""
        print(f"Training models on {X.shape[0]} samples with {X.shape[1]} features")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        results = {}
        best_score = 0
        
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            
            try:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
                
                model.fit(X_train_scaled, y_train)
                
                y_train_pred = model.predict(X_train_scaled)
                y_test_pred = model.predict(X_test_scaled)
                
                if hasattr(model, 'predict_proba'):
                    y_test_prob = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_test_prob = y_test_pred
                
                train_accuracy = accuracy_score(y_train, y_train_pred)
                test_accuracy = accuracy_score(y_test, y_test_pred)
                
                tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                try:
                    auc = roc_auc_score(y_test, y_test_prob)
                except:
                    auc = 0.5
                
                results[model_name] = {
                    'cv_accuracy_mean': cv_scores.mean(),
                    'cv_accuracy_std': cv_scores.std(),
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'auc': auc,
                    'confusion_matrix': confusion_matrix(y_test, y_test_pred)
                }
                
                print(f"  CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                print(f"  Test Accuracy: {test_accuracy:.3f}")
                print(f"  Sensitivity: {sensitivity:.3f}")
                print(f"  Specificity: {specificity:.3f}")
                
                if cv_scores.mean() > best_score:
                    best_score = cv_scores.mean()
                    self.best_model = model
                    self.best_model_name = model_name
                    
            except Exception as e:
                print(f"  Error evaluating {model_name}: {e}")
        
        print(f"\nBest model: {self.best_model_name} with CV accuracy: {best_score:.3f}")
        return results
    
    def predict_dre(self, eeg_data: np.ndarray, channel_names: List[str]) -> Dict:
        """Predict DRE for new patient."""
        if self.best_model is None:
            raise ValueError("Model not trained yet.")
        
        features = self.feature_extractor.extract_all_features(eeg_data, channel_names)
        feature_df = pd.DataFrame([features])
        
        for col in self.feature_names:
            if col not in feature_df.columns:
                feature_df[col] = 0.0
        
        feature_df = feature_df[self.feature_names]
        feature_vector_scaled = self.scaler.transform(feature_df)
        
        prediction = self.best_model.predict(feature_vector_scaled)[0]
        
        if hasattr(self.best_model, 'predict_proba'):
            probabilities = self.best_model.predict_proba(feature_vector_scaled)[0]
            probability = probabilities[1]
        else:
            probability = float(prediction)
            probabilities = [1-probability, probability]
        
        return {
            'prediction': int(prediction),
            'probability_dre': probability,
            'probabilities': probabilities,
            'model_used': self.best_model_name,
            'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low'
        }
    
    def plot_results(self, results: Dict) -> None:
        """Plot model comparison results"""
        models = []
        accuracies = []
        sensitivities = []
        specificities = []
        aucs = []
        
        for model_name, metrics in results.items():
            if 'error' not in metrics:
                models.append(model_name.replace('_', ' ').title())
                accuracies.append(metrics['test_accuracy'])
                sensitivities.append(metrics['sensitivity'])
                specificities.append(metrics['specificity'])
                aucs.append(metrics['auc'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        axes[0, 0].bar(models, accuracies, color='skyblue')
        axes[0, 0].set_title('Test Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].axhline(y=0.915, color='red', linestyle='--', label='Paper Target (91.5%)')
        axes[0, 0].legend()
        
        axes[0, 1].bar(models, sensitivities, color='lightgreen')
        axes[0, 1].set_title('Sensitivity')
        axes[0, 1].set_ylabel('Sensitivity')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].axhline(y=0.97, color='red', linestyle='--', label='Paper Target (97%)')
        axes[0, 1].legend()
        
        axes[1, 0].bar(models, specificities, color='lightcoral')
        axes[1, 0].set_title('Specificity')
        axes[1, 0].set_ylabel('Specificity')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].axhline(y=0.81, color='red', linestyle='--', label='Paper Target (81%)')
        axes[1, 0].legend()
        
        axes[1, 1].bar(models, aucs, color='gold')
        axes[1, 1].set_title('AUC')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].axhline(y=0.92, color='red', linestyle='--', label='Paper Target (0.92)')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function for EDF-based DRE prediction"""
    
    EDF_DIRECTORY = r"first package\Patient_1"
    LABELS_FILE = r"first package\Patient_1\labels.csv"
    DURATION_SECONDS = 60
    
    print("=== EDF-Based DRE Prediction ===\n")
    
    if not os.path.exists(EDF_DIRECTORY):
        print(f"ERROR: Directory not found: {EDF_DIRECTORY}")
        return
    
    predictor = DREPredictor()
    
    if not os.path.exists(LABELS_FILE):
        print("Creating labels template...")
        predictor.create_labels_template(EDF_DIRECTORY)
        print("Please edit the labels.csv file and run again.")
        return
    
    try:
        eeg_data_list, channel_names_list, labels = predictor.load_edf_files(
            EDF_DIRECTORY, LABELS_FILE, DURATION_SECONDS
        )
    except Exception as e:
        print(f"Error loading EDF files: {e}")
        return
    
    if len(eeg_data_list) == 0:
        print("No EDF files loaded successfully!")
        return
    
    print("\n=== Feature Extraction ===")
    feature_matrix = predictor.extract_features_from_data(eeg_data_list, channel_names_list)
    
    print("\n=== Model Training ===")
    results = predictor.train_and_evaluate(feature_matrix, labels)
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    for model_name, metrics in results.items():
        if 'error' not in metrics:
            print(f"{model_name}: {metrics['test_accuracy']:.3f} accuracy")
    
    print(f"\n=== Example Prediction ===")
    test_result = predictor.predict_dre(eeg_data_list[0], channel_names_list[0])
    print(f"Prediction: {'DRE' if test_result['prediction'] == 1 else 'Non-DRE'}")
    print(f"Confidence: {test_result['probability_dre']:.3f}")
    
    try:
        predictor.plot_results(results)
    except:
        print("Plotting not available")

if __name__ == "__main__":
    main()
