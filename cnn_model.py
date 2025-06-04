"""
CNN Model for DRE Prediction
Based on: "Early prediction of drug-resistant epilepsy using clinical and EEG features 
based on convolutional neural network"

This implementation creates a 7-layer CNN that processes raw EEG data
and optionally combines it with clinical features.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class EEGPreprocessor:
    """
    Preprocess EEG data for CNN input.
    Handles filtering, artifact removal, and data formatting.
    """
    
    def __init__(self, sfreq: int = 256, epoch_duration: int = 90):
        self.sfreq = sfreq
        self.epoch_duration = epoch_duration
        self.n_timepoints = sfreq * epoch_duration
        
    def bandpass_filter(self, data: np.ndarray, low_freq: float = 0.5, high_freq: float = 50.0) -> np.ndarray:
        """Apply bandpass filter to remove artifacts"""
        from scipy import signal
        
        nyquist = self.sfreq / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_data = signal.filtfilt(b, a, data, axis=-1)
        return filtered_data
    
    def remove_artifacts(self, data: np.ndarray, threshold_std: float = 4.0) -> np.ndarray:
        """Simple artifact removal based on amplitude thresholding"""
        # Calculate z-scores
        mean_val = np.mean(data, axis=-1, keepdims=True)
        std_val = np.std(data, axis=-1, keepdims=True)
        z_scores = np.abs((data - mean_val) / (std_val + 1e-10))
        
        # Replace artifacts with interpolated values
        artifact_mask = z_scores > threshold_std
        data_clean = data.copy()
        
        for ch in range(data.shape[0]):
            if np.any(artifact_mask[ch]):
                # Simple linear interpolation for artifacts
                artifact_indices = np.where(artifact_mask[ch])[0]
                clean_indices = np.where(~artifact_mask[ch])[0]
                
                if len(clean_indices) > 1:
                    data_clean[ch, artifact_indices] = np.interp(
                        artifact_indices, clean_indices, data[ch, clean_indices]
                    )
        
        return data_clean
    
    def create_epochs(self, data: np.ndarray, n_epochs: int = 10) -> np.ndarray:
        """
        Create multiple epochs from continuous EEG data.
        Paper uses 10 epochs of 90 seconds each per patient.
        
        Args:
            data: EEG data (channels x time_points)
            n_epochs: Number of epochs to create
            
        Returns:
            Epoched data (n_epochs x channels x time_points_per_epoch)
        """
        n_channels, n_timepoints = data.shape
        epoch_length = self.n_timepoints
        
        if n_timepoints < epoch_length * n_epochs:
            # If not enough data, repeat the data
            repeat_factor = int(np.ceil(epoch_length * n_epochs / n_timepoints))
            data = np.tile(data, (1, repeat_factor))
            n_timepoints = data.shape[1]
        
        epochs = []
        step_size = max(1, (n_timepoints - epoch_length) // (n_epochs - 1)) if n_epochs > 1 else 0
        
        for i in range(n_epochs):
            start_idx = i * step_size
            end_idx = start_idx + epoch_length
            
            if end_idx > n_timepoints:
                # Handle edge case
                start_idx = n_timepoints - epoch_length
                end_idx = n_timepoints
            
            epoch_data = data[:, start_idx:end_idx]
            epochs.append(epoch_data)
        
        return np.array(epochs)
    
    def preprocess_eeg(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Complete preprocessing pipeline for EEG data.
        
        Args:
            eeg_data: Raw EEG data (channels x time_points)
            
        Returns:
            Preprocessed epoched data (n_epochs x channels x time_points_per_epoch)
        """
        # 1. Bandpass filter
        filtered_data = self.bandpass_filter(eeg_data)
        
        # 2. Remove artifacts
        clean_data = self.remove_artifacts(filtered_data)
        
        # 3. Create epochs
        epoched_data = self.create_epochs(clean_data)
        
        # 4. Normalize each epoch
        for i in range(epoched_data.shape[0]):
            epoch = epoched_data[i]
            # Z-score normalization per channel
            mean_vals = np.mean(epoch, axis=1, keepdims=True)
            std_vals = np.std(epoch, axis=1, keepdims=True)
            epoched_data[i] = (epoch - mean_vals) / (std_vals + 1e-10)
        
        return epoched_data

class CNNDREModel:
    """
    CNN Model for DRE Prediction implementing the 7-layer architecture from the paper.
    """
    
    def __init__(self, n_channels: int = 19, n_timepoints: int = 23040, 
                 include_clinical: bool = True, n_clinical_features: int = 10):
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.include_clinical = include_clinical
        self.n_clinical_features = n_clinical_features
        
        self.eeg_model = None
        self.combined_model = None
        self.scaler = StandardScaler()
        self.history = None
        
    def build_eeg_model(self) -> keras.Model:
        """
        Build the 7-layer CNN model for EEG data as described in the paper.
        
        Architecture:
        1. Conv Layer 1: Kernel 5×10, ReLU
        2. Conv Layer 2: Kernel 5×10, ReLU  
        3. Flatten Layer
        4. Dense Layer 1: 128 neurons, ReLU
        5. Dense Layer 2: 64 neurons, ReLU
        6. Dense Layer 3: 32 neurons, ReLU
        7. Output Layer: 2 classes, Softmax
        """
        
        # Input layer
        eeg_input = layers.Input(shape=(self.n_channels, self.n_timepoints), name='eeg_input')
        
        # Reshape for Conv2D (add channel dimension)
        x = layers.Reshape((self.n_channels, self.n_timepoints, 1))(eeg_input)
        
        # Convolutional Layer 1
        x = layers.Conv2D(
            filters=32,
            kernel_size=(5, 10),
            activation='relu',
            padding='same',
            name='conv_layer_1'
        )(x)
        
        # Optional: Add BatchNormalization and Dropout for better training
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Convolutional Layer 2
        x = layers.Conv2D(
            filters=64,
            kernel_size=(5, 10),
            activation='relu',
            padding='same',
            name='conv_layer_2'
        )(x)
        
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Optional: Add pooling to reduce dimensionality
        x = layers.MaxPooling2D(pool_size=(2, 4))(x)
        
        # Flatten Layer
        x = layers.Flatten(name='flatten_layer')(x)
        
        # Dense Layer 1
        x = layers.Dense(128, activation='relu', name='dense_layer_1')(x)
        x = layers.Dropout(0.3)(x)
        
        # Dense Layer 2
        x = layers.Dense(64, activation='relu', name='dense_layer_2')(x)
        x = layers.Dropout(0.3)(x)
        
        # Dense Layer 3
        x = layers.Dense(32, activation='relu', name='dense_layer_3')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output Layer
        eeg_output = layers.Dense(2, activation='softmax', name='eeg_output')(x)
        
        model = keras.Model(inputs=eeg_input, outputs=eeg_output, name='EEG_CNN_Model')
        
        return model
    
    def build_combined_model(self) -> keras.Model:
        """
        Build combined model that integrates EEG CNN features with clinical features.
        This corresponds to the Clinical-EEG model from the paper.
        """
        
        # EEG branch (reuse the CNN architecture but without final classification)
        eeg_input = layers.Input(shape=(self.n_channels, self.n_timepoints), name='eeg_input')
        
        # Reshape for Conv2D
        x = layers.Reshape((self.n_channels, self.n_timepoints, 1))(eeg_input)
        
        # Convolutional layers
        x = layers.Conv2D(32, (5, 10), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Conv2D(64, (5, 10), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.MaxPooling2D(pool_size=(2, 4))(x)
        
        # Flatten and dense layers
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        eeg_features = layers.Dropout(0.3)(x)
        
        # Clinical features branch
        clinical_input = layers.Input(shape=(self.n_clinical_features,), name='clinical_input')
        clinical_features = layers.Dense(32, activation='relu')(clinical_input)
        clinical_features = layers.Dropout(0.2)(clinical_features)
        
        # Combine EEG and clinical features
        combined_features = layers.concatenate([eeg_features, clinical_features])
        
        # Final classification layers
        x = layers.Dense(32, activation='relu')(combined_features)
        x = layers.Dropout(0.2)(x)
        output = layers.Dense(2, activation='softmax', name='combined_output')(x)
        
        model = keras.Model(
            inputs=[eeg_input, clinical_input], 
            outputs=output, 
            name='Combined_EEG_Clinical_Model'
        )
        
        return model
    
    def compile_model(self, model: keras.Model, learning_rate: float = 0.001) -> None:
        """Compile the model with appropriate optimizer and loss function"""
        
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',  # For integer labels
            metrics=['accuracy', 'precision', 'recall']
        )
    
    def create_callbacks(self, patience: int = 10) -> List[callbacks.Callback]:
        """Create callbacks for training"""
        
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callback_list
    
    def train_eeg_model(self, X_eeg: np.ndarray, y: np.ndarray, 
                       validation_split: float = 0.2, epochs: int = 100, 
                       batch_size: int = 32) -> Dict:
        """
        Train the EEG-only CNN model.
        
        Args:
            X_eeg: EEG data (n_samples x n_channels x n_timepoints)
            y: Labels (n_samples,)
            validation_split: Fraction of data for validation
            epochs: Maximum number of training epochs
            batch_size: Training batch size
            
        Returns:
            Training history and metrics
        """
        
        print("Building EEG CNN model...")
        self.eeg_model = self.build_eeg_model()
        self.compile_model(self.eeg_model)
        
        print(f"Model architecture:")
        self.eeg_model.summary()
        
        # Create callbacks
        callback_list = self.create_callbacks()
        
        print(f"Training EEG model on {X_eeg.shape[0]} samples...")
        
        # Train the model
        history = self.eeg_model.fit(
            X_eeg, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        self.history = history
        
        return history.history
    
    def train_combined_model(self, X_eeg: np.ndarray, X_clinical: np.ndarray, 
                           y: np.ndarray, validation_split: float = 0.2, 
                           epochs: int = 100, batch_size: int = 32) -> Dict:
        """
        Train the combined EEG + Clinical CNN model.
        
        Args:
            X_eeg: EEG data (n_samples x n_channels x n_timepoints)
            X_clinical: Clinical features (n_samples x n_clinical_features)
            y: Labels (n_samples,)
            validation_split: Fraction of data for validation
            epochs: Maximum number of training epochs
            batch_size: Training batch size
            
        Returns:
            Training history and metrics
        """
        
        # Normalize clinical features
        X_clinical_scaled = self.scaler.fit_transform(X_clinical)
        
        print("Building combined EEG + Clinical CNN model...")
        self.combined_model = self.build_combined_model()
        self.compile_model(self.combined_model)
        
        print(f"Model architecture:")
        self.combined_model.summary()
        
        # Create callbacks
        callback_list = self.create_callbacks()
        
        print(f"Training combined model on {X_eeg.shape[0]} samples...")
        
        # Train the model
        history = self.combined_model.fit(
            [X_eeg, X_clinical_scaled], y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        self.history = history
        
        return history.history
    
    def evaluate_model(self, X_eeg: np.ndarray, X_clinical: Optional[np.ndarray], 
                      y: np.ndarray, model_type: str = 'eeg') -> Dict:
        """
        Evaluate the trained model.
        
        Args:
            X_eeg: EEG test data
            X_clinical: Clinical test data (optional)
            y: True labels
            model_type: 'eeg' or 'combined'
            
        Returns:
            Dictionary with evaluation metrics
        """
        
        if model_type == 'eeg' and self.eeg_model is not None:
            model = self.eeg_model
            predictions = model.predict(X_eeg)
        elif model_type == 'combined' and self.combined_model is not None:
            model = self.combined_model
            X_clinical_scaled = self.scaler.transform(X_clinical)
            predictions = model.predict([X_eeg, X_clinical_scaled])
        else:
            raise ValueError(f"Model type '{model_type}' not available or not trained")
        
        # Convert predictions to class labels
        y_pred = np.argmax(predictions, axis=1)
        y_prob = predictions[:, 1]  # Probability of DRE class
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # AUC
        auc = roc_auc_score(y, y_prob)
        
        results = {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'auc': auc,
            'confusion_matrix': confusion_matrix(y, y_pred),
            'predictions': y_pred,
            'probabilities': y_prob,
            'classification_report': classification_report(y, y_pred)
        }
        
        return results
    
    def predict(self, X_eeg: np.ndarray, X_clinical: Optional[np.ndarray] = None, 
               model_type: str = 'eeg') -> Dict:
        """
        Make predictions for new patients.
        
        Args:
            X_eeg: EEG data for prediction
            X_clinical: Clinical data (optional)
            model_type: 'eeg' or 'combined'
            
        Returns:
            Dictionary with predictions and probabilities
        """
        
        if model_type == 'eeg' and self.eeg_model is not None:
            predictions = self.eeg_model.predict(X_eeg)
        elif model_type == 'combined' and self.combined_model is not None:
            X_clinical_scaled = self.scaler.transform(X_clinical)
            predictions = self.combined_model.predict([X_eeg, X_clinical_scaled])
        else:
            raise ValueError(f"Model type '{model_type}' not available or not trained")
        
        y_pred = np.argmax(predictions, axis=1)
        y_prob = predictions[:, 1]
        
        results = {
            'predictions': y_pred,
            'probabilities': y_prob,
            'risk_levels': ['High' if p > 0.7 else 'Medium' if p > 0.3 else 'Low' for p in y_prob]
        }
        
        return results
    
    def plot_training_history(self) -> None:
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        history = self.history.history
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(history['loss'], label='Training Loss')
        axes[0, 1].plot(history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Precision
        if 'precision' in history:
            axes[1, 0].plot(history['precision'], label='Training Precision')
            axes[1, 0].plot(history['val_precision'], label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
        
        # Recall
        if 'recall' in history:
            axes[1, 1].plot(history['recall'], label='Training Recall')
            axes[1, 1].plot(history['val_recall'], label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()

def generate_synthetic_cnn_data(n_patients: int = 101, n_channels: int = 19, 
                               epoch_duration: int = 90, sfreq: int = 256, 
                               n_epochs_per_patient: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic data for CNN training.
    Creates data similar to the paper: 101 patients, 10 epochs each, 90 seconds per epoch.
    """
    
    print(f"Generating synthetic CNN data for {n_patients} patients...")
    
    n_timepoints = epoch_duration * sfreq
    total_epochs = n_patients * n_epochs_per_patient
    
    # Generate EEG data
    X_eeg = np.zeros((total_epochs, n_channels, n_timepoints))
    
    # Generate clinical features
    clinical_features = [
        'age', 'sex', 'seizure_frequency', 'seizure_duration', 'family_history',
        'mri_abnormal', 'eeg_abnormal', 'medication_response', 'comorbidities', 'onset_age'
    ]
    X_clinical = np.zeros((total_epochs, len(clinical_features)))
    
    # Labels
    y = np.zeros(total_epochs, dtype=int)
    
    # Generate data for each patient
    epoch_idx = 0
    for patient_id in range(n_patients):
        # Determine if patient is DRE (first third are DRE)
        is_dre = patient_id < n_patients // 3
        
        # Generate clinical features for this patient
        if is_dre:
            # DRE patients tend to have certain characteristics
            patient_clinical = np.array([
                np.random.normal(35, 10),  # age
                np.random.binomial(1, 0.6),  # sex (slightly more female)
                np.random.exponential(10) + 5,  # higher seizure frequency
                np.random.exponential(2) + 1,  # longer seizure duration
                np.random.binomial(1, 0.4),  # family history
                np.random.binomial(1, 0.7),  # MRI abnormal (higher chance)
                np.random.binomial(1, 0.8),  # EEG abnormal (higher chance)
                np.random.uniform(0, 0.3),  # poor medication response
                np.random.binomial(1, 0.5),  # comorbidities
                np.random.normal(15, 8)  # earlier onset age
            ])
        else:
            # Non-DRE patients
            patient_clinical = np.array([
                np.random.normal(40, 12),  # age
                np.random.binomial(1, 0.5),  # sex
                np.random.exponential(3) + 1,  # lower seizure frequency
                np.random.exponential(1) + 0.5,  # shorter seizure duration
                np.random.binomial(1, 0.2),  # family history
                np.random.binomial(1, 0.3),  # MRI abnormal (lower chance)
                np.random.binomial(1, 0.5),  # EEG abnormal (lower chance)
                np.random.uniform(0.7, 1.0),  # good medication response
                np.random.binomial(1, 0.2),  # comorbidities
                np.random.normal(25, 10)  # later onset age
            ])
        
        # Generate EEG epochs for this patient
        for epoch in range(n_epochs_per_patient):
            # Base EEG signal
            eeg_epoch = np.random.randn(n_channels, n_timepoints) * 0.5
            
            # Add realistic EEG characteristics
            t = np.linspace(0, epoch_duration, n_timepoints)
            
            # Add frequency-specific content
            for ch in range(n_channels):
                # Alpha rhythm (8-12 Hz)
                alpha_amp = 1.0 if ch >= n_channels//2 else 0.5  # Stronger in posterior
                eeg_epoch[ch] += alpha_amp * np.sin(2 * np.pi * 10 * t + np.random.uniform(0, 2*np.pi))
                
                # Theta rhythm (4-8 Hz) - important for DRE
                if is_dre:
                    # Stronger, more synchronized theta in DRE patients
                    theta_amp = 2.0
                    theta_phase = np.random.uniform(0, np.pi/2)  # More synchronized
                else:
                    theta_amp = 1.0
                    theta_phase = np.random.uniform(0, 2*np.pi)  # Random phase
                
                eeg_epoch[ch] += theta_amp * np.sin(2 * np.pi * 6 * t + theta_phase)
                
                # Beta rhythm (13-30 Hz)
                beta_amp = 0.5
                eeg_epoch[ch] += beta_amp * np.sin(2 * np.pi * 20 * t + np.random.uniform(0, 2*np.pi))
            
            # Add some epileptiform activity for DRE patients
            if is_dre and np.random.random() < 0.3:  # 30% chance of spikes
                # Add spike-like activity
                spike_time = np.random.randint(n_timepoints//4, 3*n_timepoints//4)
                spike_channels = np.random.choice(n_channels, size=np.random.randint(1, 4), replace=False)
                
                for spike_ch in spike_channels:
                    # Create spike waveform
                    spike_duration = int(0.1 * sfreq)  # 100ms spike
                    spike_start = max(0, spike_time - spike_duration//2)
                    spike_end = min(n_
