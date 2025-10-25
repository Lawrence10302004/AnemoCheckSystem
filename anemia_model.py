"""
Anemia CBC Model Module
----------------------
This module handles the machine learning model for anemia classification based on CBC data.
"""

import os
import io
import base64
import logging
import numpy as np
import pandas as pd
import matplotlib
# Use non-interactive backend to avoid GUI/Tkinter initialization when generating
# figures from a web server or background thread.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

logger = logging.getLogger(__name__)


class AnemiaCBCModel:
    """
    Anemia classification model based on WHO guidelines using CBC data.
    
    This class handles the training, evaluation, and prediction for the anemia
    classification model based on hemoglobin levels according to WHO guidelines.
    """
    
    def __init__(self):
        """Initialize the anemia model with default parameters."""
        # Thresholds based on WHO guidelines
        self.threshold_normal = 12.0    # >= 12 g/dL: Normal
        self.threshold_mild = 10.0      # 10-11.9 g/dL: Mild anemia
        self.threshold_moderate = 8.0   # 8-9.9 g/dL: Moderate anemia
                                        # < 8 g/dL: Severe anemia
        
        # Model parameters
        self.model_type = 'decision_tree'
        self.model = None
        self.feature_names = ['Hemoglobin (g/dL)']
        self.class_names = ['Normal', 'Mild', 'Moderate', 'Severe']
        
        # Medical recommendations
        self.recommendations = {
            'Normal': "Maintain a healthy diet rich in iron, vitamin B12, and folate.",
            'Mild': "Consider dietary adjustments to increase iron intake and monitor hemoglobin "
                   "levels in 1-2 months. Foods rich in iron include red meat, spinach, and legumes.",
            'Moderate': "Medical consultation recommended. Iron supplements may be prescribed. "
                       "Further testing might be needed to determine the underlying cause.",
            'Severe': "Emergency medical care required. Immediate consultation with a healthcare "
                     "provider is necessary as severe anemia can lead to serious complications."
        }
    
    def update_thresholds(self, threshold_normal, threshold_mild, threshold_moderate):
        """
        Update the hemoglobin thresholds for classification.
        
        Parameters:
        -----------
        threshold_normal : float
            Hemoglobin threshold for normal (g/dL)
        threshold_mild : float
            Hemoglobin threshold for mild anemia (g/dL)
        threshold_moderate : float
            Hemoglobin threshold for moderate anemia (g/dL)
        """
        self.threshold_normal = threshold_normal
        self.threshold_mild = threshold_mild
        self.threshold_moderate = threshold_moderate
        
        # Re-initialize the model with new thresholds
        if self.model is not None:
            self.initialize()
    
    def set_model_type(self, model_type):
        """
        Set the model type ('decision_tree' or 'random_forest').
        
        Parameters:
        -----------
        model_type : str
            Model type ('decision_tree' or 'random_forest')
        """
        if model_type not in ['decision_tree', 'random_forest']:
            raise ValueError("Model type must be 'decision_tree' or 'random_forest'")
        
        self.model_type = model_type
        
        # Re-initialize the model with new type
        if self.model is not None:
            self.initialize()
    
    def generate_synthetic_data(self, n_samples=1000, random_seed=42):
        """
        Generate synthetic hemoglobin data and classify according to WHO guidelines.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        random_seed : int
            Random seed for reproducibility
        
        Returns:
        --------
        X : pandas.DataFrame
            Features (hemoglobin values)
        y : pandas.Series
            Anemia severity labels
        """
        np.random.seed(random_seed)
        
        # Generate hemoglobin values with realistic distribution
        # Normal distribution around 13 g/dL with standard deviation 2.5
        hemoglobin = np.random.normal(loc=13, scale=2.5, size=n_samples)
        
        # Clip values to be within realistic range (4-18 g/dL)
        hemoglobin = np.clip(hemoglobin, 4, 18)
        
        # Create DataFrame
        X = pd.DataFrame({'hemoglobin': hemoglobin})
        
        # Classify according to WHO guidelines
        y = pd.Series(index=X.index, name='anemia_class', dtype='object')
        
        # Apply classification rules based on current thresholds
        y[X['hemoglobin'] >= self.threshold_normal] = 'Normal'
        y[(X['hemoglobin'] >= self.threshold_mild) & (X['hemoglobin'] < self.threshold_normal)] = 'Mild'
        y[(X['hemoglobin'] >= self.threshold_moderate) & (X['hemoglobin'] < self.threshold_mild)] = 'Moderate'
        y[X['hemoglobin'] < self.threshold_moderate] = 'Severe'
        
        return X, y
    
    def initialize(self):
        """Initialize the model by training on synthetic data."""
        logger.info("Initializing anemia classification model...")
        
        # Generate synthetic data
        X, y = self.generate_synthetic_data(n_samples=1000)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create and train the model
        if self.model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(
                max_depth=3,
                criterion='entropy',
                random_state=42,
                class_weight='balanced'
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                class_weight='balanced'
            )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        logger.info(f"Model initialized with accuracy: {accuracy:.4f}")
        logger.debug(f"Classification report:\n{report}")
    
    def predict(self, hemoglobin):
        """
        Make a prediction based on hemoglobin value.
        
        Parameters:
        -----------
        hemoglobin : float
            Hemoglobin value in g/dL
        
        Returns:
        --------
        result : dict
            Dictionary containing prediction and recommendation
        """
        # Initialize model if not already done
        if self.model is None:
            self.initialize()
        
        # Direct classification based on thresholds
        if hemoglobin >= self.threshold_normal:
            predicted_class = 'Normal'
        elif hemoglobin >= self.threshold_mild:
            predicted_class = 'Mild'
        elif hemoglobin >= self.threshold_moderate:
            predicted_class = 'Moderate'
        else:
            predicted_class = 'Severe'
        
        # Also get model prediction (can be different due to other factors in a real model)
        X_pred = np.array([[hemoglobin]])
        model_class = self.model.predict(X_pred)[0]
        
        # Get probability estimates from model
        probabilities = self.model.predict_proba(X_pred)[0]
        class_names = self.model.classes_
        prob_dict = {class_name: prob for class_name, prob in zip(class_names, probabilities)}
        
        # Use threshold-based classification, but include model probabilities
        confidence = prob_dict.get(predicted_class, 0.0)
        
        # Create result dictionary
        result = {
            'hemoglobin': hemoglobin,
            'predicted_class': predicted_class,
            'model_class': model_class,
            'confidence': confidence,
            'recommendation': self.recommendations[predicted_class],
            'all_probabilities': prob_dict,
            'thresholds': {
                'normal': self.threshold_normal,
                'mild': self.threshold_mild,
                'moderate': self.threshold_moderate
            }
        }
        
        return result
    
    def get_tree_visualization(self):
        """
        Generate a decision tree visualization image (if model is decision tree).
        
        Returns:
        --------
        str or None
            Base64-encoded PNG image of decision tree, or None if not applicable
        """
        if self.model is None:
            self.initialize()
        
        if not isinstance(self.model, DecisionTreeClassifier):
            return None
        
        plt.figure(figsize=(12, 8))
        plot_tree(
            self.model,
            feature_names=self.feature_names,
            class_names=self.class_names,
            filled=True,
            rounded=True,
            proportion=False,
            precision=2
        )
        
        # Add WHO guidelines annotation
        guideline_text = (
            f"WHO Anemia Guidelines:\n"
            f"• Normal: Hemoglobin ≥ {self.threshold_normal} g/dL\n"
            f"• Mild: Hemoglobin {self.threshold_mild}-{self.threshold_normal-0.1} g/dL\n"
            f"• Moderate: Hemoglobin {self.threshold_moderate}-{self.threshold_mild-0.1} g/dL\n"
            f"• Severe: Hemoglobin < {self.threshold_moderate} g/dL"
        )
        plt.annotate(
            guideline_text,
            xy=(0.05, 0.05),
            xycoords='figure fraction',
            bbox=dict(boxstyle="round,pad=0.5", fc="lightgray", alpha=0.8)
        )
        
        plt.title("Decision Tree for Anemia Classification", fontsize=14)
        plt.tight_layout()
        
        # Save to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # Encode to base64 for embedding in HTML
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return f"data:image/png;base64,{img_str}"