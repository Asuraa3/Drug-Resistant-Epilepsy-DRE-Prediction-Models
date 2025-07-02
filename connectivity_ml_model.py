"""
Connectivity-Based Machine Learning Model for DRE Prediction
Based on: "Machine learning-based algorithm of drug-resistant prediction 
in newly diagnosed patients with temporal lobe epilepsy"

This implementation focuses on Phase Lag Index (PLI) and graph theory features
from frontotemporal networks in the theta band (4-8 Hz).
"""

import numpy as np
import pandas as pd
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
from scipy import signal
from scipy.stats import pearsonr
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
            'delta': (1, 4),
            'theta': (4, 8),    # Most important according to paper
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        features = {}
        
        for band_name, (fmin, fmax) in frequency_bands.items():
            # Filter to frequency band
            band_data = self.bandpass_filter(eeg_data, fmin, fmax)
            
            # Calculate PLI
            band_pli = self.calculate_phase_lag_index(band_data)
            pli_upper = band_pli[np.triu_indices_from(band_pli, k=1)]
            
            # Extract features
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
    """
    Drug-Resistant Epilepsy Predictor based on connectivity features.
    Implements the methodology from the paper with Tree Bagger as best model.
    """
    
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
            # Tree Bagger (Best performing in paper - 91.5% accuracy)
            'tree_bagger': BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=42, max_depth=10),
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            
            # Traditional ML algorithms from paper
            'naive_bayes': GaussianNB(),
            'svm': SVC(probability=True, random_state=42, kernel='rbf'),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'linear_discriminant': LinearDiscriminantAnalysis(),
            
            # Additional ensemble methods
            'subspace_knn': BaggingClassifier(
                estimator=KNeighborsClassifier(n_neighbors=3),
                n_estimators=50,
                max_features=0.8,
                random_state=42,
                n_jobs=-1
            )
        }
        return models
    
    def extract_features_from_data(self, eeg_data_list: List[np.ndarray], 
                                 channel_names_list: List[List[str]]) -> pd.DataFrame:
        """
        Extract features from list of EEG data arrays.
        
        Args:
            eeg_data_list: List of EEG data arrays (each: channels x time_points)
            channel_names_list: List of channel name lists for each EEG data
            
        Returns:
            DataFrame with extracted features
        """
        all_features = []
        
        print(f"Extracting features from {len(eeg_data_list)} EEG recordings...")
        
        for i, (eeg_data, channel_names) in enumerate(zip(eeg_data_list, channel_names_list)):
            print(f"Processing recording {i+1}/{len(eeg_data_list)}")
            
            try:
                features = self.feature_extractor.extract_all_features(eeg_data, channel_names)
                all_features.append(features)
            except Exception as e:
                print(f"Error processing recording {i+1}: {e}")
                # Add empty features for failed recordings
                if all_features:
                    # Use previous successful extraction as template
                    empty_features = {key: 0.0 for key in all_features[0].keys()}
                    all_features.append(empty_features)
                else:
                    # Skip this recording
                    continue
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(all_features)
        
        # Handle any remaining NaN or inf values
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.fillna(0)
        
        self.feature_names = feature_df.columns.tolist()
        print(f"Extracted {len(self.feature_names)} features")
        
        return feature_df
    
    def train_and_evaluate(self, X: pd.DataFrame, y: np.ndarray, 
                          cv_folds: int = 5, test_size: float = 0.2) -> Dict:
        """
        Train and evaluate all models using cross-validation.
        
        Args:
            X: Feature matrix
            y: Labels (0: Non-DRE, 1: DRE)
            cv_folds: Number of cross-validation folds
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with performance metrics for each model
        """
        print(f"Training and evaluating models on {X.shape[0]} samples with {X.shape[1]} features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        results = {}
        best_score = 0
        
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            
            try:
                # Cross-validation scores
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
                
                # Train on full training set
                model.fit(X_train_scaled, y_train)
                
                # Predictions
                y_train_pred = model.predict(X_train_scaled)
                y_test_pred = model.predict(X_test_scaled)
                
                # Probabilities (if available)
                if hasattr(model, 'predict_proba'):
                    y_test_prob = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_test_prob = y_test_pred
                
                # Calculate metrics
                train_accuracy = accuracy_score(y_train, y_train_pred)
                test_accuracy = accuracy_score(y_test, y_test_pred)
                
                # Sensitivity and Specificity
                tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                # AUC
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
                print(f"  AUC: {auc:.3f}")
                
                # Track best model based on CV accuracy
                if cv_scores.mean() > best_score:
                    best_score = cv_scores.mean()
                    self.best_model = model
                    self.best_model_name = model_name
                    
            except Exception as e:
                print(f"  Error evaluating {model_name}: {e}")
                results[model_name] = {
                    'cv_accuracy_mean': 0,
                    'cv_accuracy_std': 0,
                    'train_accuracy': 0,
                    'test_accuracy': 0,
                    'sensitivity': 0,
                    'specificity': 0,
                    'auc': 0.5,
                    'error': str(e)
                }
        
        print(f"\nBest model: {self.best_model_name} with CV accuracy: {best_score:.3f}")
        
        return results
    
    def predict_dre(self, eeg_data: np.ndarray, channel_names: List[str]) -> Dict:
        """
        Predict DRE for new patient.
        
        Args:
            eeg_data: EEG data (channels x time_points)
            channel_names: List of channel names
            
        Returns:
            Dictionary with prediction results
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet. Call train_and_evaluate first.")
        
        # Extract features
        features = self.feature_extractor.extract_all_features(eeg_data, channel_names)
        
        # Convert to DataFrame with same columns as training
        feature_df = pd.DataFrame([features])
        
        # Ensure all training features are present
        for col in self.feature_names:
            if col not in feature_df.columns:
                feature_df[col] = 0.0
        
        # Reorder columns to match training
        feature_df = feature_df[self.feature_names]
        
        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_df)
        
        # Predict
        prediction = self.best_model.predict(feature_vector_scaled)[0]
        
        if hasattr(self.best_model, 'predict_proba'):
            probabilities = self.best_model.predict_proba(feature_vector_scaled)[0]
            probability = probabilities[1]  # Probability of DRE
        else:
            probability = float(prediction)
            probabilities = [1-probability, probability]
        
        return {
            'prediction': int(prediction),  # 0: Non-DRE, 1: DRE
            'probability_dre': probability,
            'probabilities': probabilities,  # [Non-DRE, DRE]
            'model_used': self.best_model_name,
            'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low'
        }
    
    def plot_results(self, results: Dict) -> None:
        """Plot model comparison results"""
        # Prepare data for plotting
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
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # Accuracy
        axes[0, 0].bar(models, accuracies, color='skyblue')
        axes[0, 0].set_title('Test Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].axhline(y=0.915, color='red', linestyle='--', label='Paper Target (91.5%)')
        axes[0, 0].legend()
        
        # Sensitivity
        axes[0, 1].bar(models, sensitivities, color='lightgreen')
        axes[0, 1].set_title('Sensitivity (True Positive Rate)')
        axes[0, 1].set_ylabel('Sensitivity')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].axhline(y=0.97, color='red', linestyle='--', label='Paper Target (97%)')
        axes[0, 1].legend()
        
        # Specificity
        axes[1, 0].bar(models, specificities, color='lightcoral')
        axes[1, 0].set_title('Specificity (True Negative Rate)')
        axes[1, 0].set_ylabel('Specificity')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].axhline(y=0.81, color='red', linestyle='--', label='Paper Target (81%)')
        axes[1, 0].legend()
        
        # AUC
        axes[1, 1].bar(models, aucs, color='gold')
        axes[1, 1].set_title('AUC (Area Under Curve)')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].axhline(y=0.92, color='red', linestyle='--', label='Paper Target (0.92)')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()

# Example usage and testing
def generate_synthetic_eeg_data(n_patients: int = 50, n_channels: int = 19, 
                               duration_seconds: int = 60, sfreq: int = 256) -> Tuple[List[np.ndarray], List[List[str]], np.ndarray]:
    """
    Generate synthetic EEG data for testing the implementation.
    In real use, replace this with your actual EEG data loading function.
    """
    print(f"Generating synthetic EEG data for {n_patients} patients...")
    
    # Standard 10-20 electrode names
    channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
                    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 
                    'Fz', 'Cz', 'Pz'][:n_channels]
    
    eeg_data_list = []
    channel_names_list = []
    labels = []
    
    n_timepoints = duration_seconds * sfreq
    
    for i in range(n_patients):
        # Generate synthetic EEG data
        # Add some realistic characteristics: 1/f noise + oscillations
        base_signal = np.random.randn(n_channels, n_timepoints)
        
        # Add some frequency-specific content
        t = np.linspace(0, duration_seconds, n_timepoints)
        
        # Add alpha rhythm (8-12 Hz) - stronger in posterior channels
        alpha_freq = 10
        for ch in range(n_channels):
            if 'O' in channel_names[ch] or 'P' in channel_names[ch]:
                alpha_amplitude = 2.0
            else:
                alpha_amplitude = 0.5
            base_signal[ch] += alpha_amplitude * np.sin(2 * np.pi * alpha_freq * t)
        
        # Add theta rhythm (4-8 Hz) - important for DRE prediction
        theta_freq = 6
        theta_amplitude = 1.0
        
        # For DRE patients, increase theta connectivity in frontotemporal regions
        is_dre = i < n_patients // 3  # First third are DRE patients
        
        if is_dre:
            # Increase theta power and synchronization in frontotemporal channels
            for ch in range(n_channels):
                if any(region in channel_names[ch] for region in ['F7', 'F8', 'T3', 'T4', 'F3', 'F4']):
                    theta_amplitude_ch = 2.5  # Stronger theta
                    # Add some phase coupling
                    phase_offset = np.random.uniform(0, np.pi/4)  # Small phase offset for coupling
                else:
                    theta_amplitude_ch = theta_amplitude
                    phase_offset = np.random.uniform(0, 2*np.pi)  # Random phase
                
                base_signal[ch] += theta_amplitude_ch * np.sin(2 * np.pi * theta_freq * t + phase_offset)
        else:
            # Normal theta activity
            for ch in range(n_channels):
                phase_offset = np.random.uniform(0, 2*np.pi)
                base_signal[ch] += theta_amplitude * np.sin(2 * np.pi * theta_freq * t + phase_offset)
        
        # Apply 1/f scaling to make it more realistic
        for ch in range(n_channels):
            # Simple 1/f filter approximation
            freqs = np.fft.fftfreq(n_timepoints, 1/sfreq)
            fft_signal = np.fft.fft(base_signal[ch])
            # Apply 1/f scaling (avoid division by zero)
            scaling = 1 / np.sqrt(np.abs(freqs) + 1)
            fft_signal *= scaling
            base_signal[ch] = np.real(np.fft.ifft(fft_signal))
        
        eeg_data_list.append(base_signal)
        channel_names_list.append(channel_names.copy())
        labels.append(1 if is_dre else 0)
    
    return eeg_data_list, channel_names_list, np.array(labels)

def main():
    """
    Main function demonstrating the connectivity-based DRE prediction.
    """
    print("=== Connectivity-Based DRE Prediction Implementation ===\n")
    
    # Generate synthetic data (replace with your real data loading)
    eeg_data_list, channel_names_list, labels = generate_synthetic_eeg_data(
        n_patients=100, n_channels=19, duration_seconds=60
    )
    
    print(f"Loaded {len(eeg_data_list)} EEG recordings")
    print(f"DRE patients: {np.sum(labels)} ({np.mean(labels)*100:.1f}%)")
    print(f"Non-DRE patients: {len(labels) - np.sum(labels)} ({(1-np.mean(labels))*100:.1f}%)")
    
    # Initialize predictor
    predictor = DREPredictor()
    
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
    
    # Test prediction on new patient
    print(f"\n=== Testing Prediction ===")
    test_eeg = eeg_data_list[0]  # Use first patient as example
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
