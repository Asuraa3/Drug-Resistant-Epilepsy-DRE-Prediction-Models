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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks

"""
per le serie temporali, dove bisogna passare la dimensione del dataset, la lunghezza della finestra temporale e la 
dimensione del dataset contenuta in essa.
"""


class CNNDREModel:
    """
    CNN Model for DRE Prediction implementing the 7-layer architecture from the paper.
    """
    
    def __init__(self, n_channels: int = 19, n_timepoints: int = 23040, batch_size=50):
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.batch_size = batch_size
        self.eeg_model = None
        self.combined_model = None
        self.scaler = StandardScaler()
        self.history = None

#model to fix, not working
    def build_eeg_model(self) -> keras.Model:
        # Input shape: (n_channels, n_timepoints)
        eeg_input = layers.Input(shape=(self.n_channels, self.n_timepoints), name='eeg_input')

        # Add channel dimension: (n_channels, n_timepoints, 1)
        x = layers.Reshape((self.n_timepoints, self.n_channels, 1))(eeg_input)

        # Conv Block 1
        x = layers.Conv2D(filters=32, kernel_size=(3, 11), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(1, 4))(x)  # Pool only over time
        x = layers.Dropout(0.2)(x)

        # Conv Block 2
        x = layers.Conv2D(filters=64, kernel_size=(3, 9), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(1, 4))(x)
        x = layers.Dropout(0.3)(x)

        # Conv Block 3
        x = layers.Conv2D(filters=128, kernel_size=(3, 7), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(1, 2))(x)
        x = layers.Dropout(0.4)(x)

        # Global average pooling instead of flatten
        x = layers.GlobalAveragePooling2D()(x)

        # Fully connected layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        # Output layer (2 classes)
        eeg_output = layers.Dense(2, activation='softmax', name='eeg_output')(x)

        model = keras.Model(inputs=eeg_input, outputs=eeg_output, name='EEG_CNN_Model')

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
                       X_val, y_val, epochs: int = 100) -> Dict:
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
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        self.history = history

        self.eeg_model.save("model.keras")
        
        return history.history

    
    def evaluate_model(self, X_eeg: np.ndarray,
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
        
        if self.eeg_model is not None:
            model = self.eeg_model
            predictions = model.predict(X_eeg)
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