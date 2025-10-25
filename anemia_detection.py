#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    classification_report
)
import joblib
import logging
from model_utils import predict_anemia
from visualization import (
    plot_class_distribution, 
    plot_confusion_matrix, 
    plot_roc_curve, 
    plot_feature_importance
)

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data(file_path):
    """
    Load the dataset and preprocess it for anemia detection model.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing CBC data
    
    Returns:
    --------
    X : pandas.DataFrame
        Features for the model
    y : pandas.Series
        Target variable (0 for Normal, 1 for Anemia)
    feature_names : list
        Names of the features used
    """
    try:
        # Load the dataset
        logging.info(f"Loading dataset from {file_path}")
        df = pd.read_csv(file_path)
        
        # Display basic information about the dataset
        logging.info(f"Dataset shape: {df.shape}")
        logging.info("\nFirst few rows:")
        logging.debug(df.head())
        
        # Check for missing values
        logging.info("\nChecking for missing values:")
        missing_values = df.isnull().sum()
        logging.debug(missing_values)
        
        if missing_values.sum() > 0:
            logging.warning(f"Found {missing_values.sum()} missing values. Handling them...")
            # Fill missing values with median for numerical columns
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64'] and df[col].isnull().sum() > 0:
                    median_value = df[col].median()
                    df[col].fillna(median_value, inplace=True)
                    logging.info(f"Filled missing values in {col} with median: {median_value}")
        
        # Convert categorical labels to numerical
        logging.info("Converting labels: 'Normal' -> 0, 'Anemia' -> 1")
        df['Label'] = df['Label'].map({'Normal': 0, 'Anemia': 1})
        
        # Select relevant features
        features = ['HGB', 'RBC', 'HCT', 'MCV']
        logging.info(f"Selected features: {features}")
        
        # Extract features and target
        X = df[features]
        y = df['Label']
        
        # Display class distribution
        logging.info("\nClass distribution:")
        class_counts = y.value_counts()
        logging.debug(class_counts)
        
        return X, y, features
        
    except Exception as e:
        logging.error(f"Error in loading and preprocessing data: {e}")
        raise

def train_and_evaluate_models(X, y, feature_names):
    """
    Train and evaluate logistic regression and random forest models
    for anemia detection.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Features for the model
    y : pandas.Series
        Target variable (0 for Normal, 1 for Anemia)
    feature_names : list
        Names of the features used
    
    Returns:
    --------
    best_model : sklearn estimator
        The best performing model
    scaler : sklearn.preprocessing.StandardScaler
        The scaler used to normalize features
    """
    try:
        # Split the data into training and testing sets (80% / 20%)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logging.info(f"Data split: Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Logistic Regression model with class weights
        logging.info("\nTraining Logistic Regression model...")
        lr_model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        lr_model.fit(X_train_scaled, y_train)
        
        # Train Random Forest model
        logging.info("Training Random Forest model...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced', 
            random_state=42
        )
        rf_model.fit(X_train_scaled, y_train)
        
        # Evaluate models
        models = {
            'Logistic Regression': lr_model,
            'Random Forest': rf_model
        }
        
        best_f1 = 0
        best_model = None
        
        for name, model in models.items():
            logging.info(f"\nEvaluating {name} model...")
            
            # Predictions on training set
            y_train_pred = model.predict(X_train_scaled)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            train_f1 = f1_score(y_train, y_train_pred)
            
            # Predictions on test set
            y_test_pred = model.predict(X_test_scaled)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred)
            
            # Probability predictions for ROC curve
            y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Display results
            logging.info(f"{name} Training Accuracy: {train_accuracy:.4f}")
            logging.info(f"{name} Training F1 Score: {train_f1:.4f}")
            logging.info(f"{name} Test Accuracy: {test_accuracy:.4f}")
            logging.info(f"{name} Test F1 Score: {test_f1:.4f}")
            
            # Detailed classification report
            logging.info(f"\nClassification Report for {name}:")
            logging.info("\n" + classification_report(y_test, y_test_pred))
            
            # Generate and display confusion matrix
            cm = confusion_matrix(y_test, y_test_pred)
            plot_confusion_matrix(cm, name)
            
            # Generate and display ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_test_proba)
            roc_auc = auc(fpr, tpr)
            plot_roc_curve(fpr, tpr, roc_auc, name)
            
            # Update best model if current model has better F1 score
            if test_f1 > best_f1:
                best_f1 = test_f1
                best_model = model
        
        # Plot feature importance for Random Forest
        if 'Random Forest' in models:
            plot_feature_importance(models['Random Forest'], feature_names)
            
        logging.info(f"\nBest model based on F1 score: {best_model.__class__.__name__}")
        return best_model, scaler
    
    except Exception as e:
        logging.error(f"Error in training and evaluating models: {e}")
        raise

def save_model(model, scaler, output_dir='.'):
    """
    Save the trained model and scaler to disk.
    
    Parameters:
    -----------
    model : sklearn estimator
        The trained model to save
    scaler : sklearn.preprocessing.StandardScaler
        The scaler used to normalize features
    output_dir : str
        Directory to save the model files
    """
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save model
        model_path = os.path.join(output_dir, 'anemia_model.joblib')
        joblib.dump(model, model_path)
        logging.info(f"Model saved to {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(output_dir, 'anemia_scaler.joblib')
        joblib.dump(scaler, scaler_path)
        logging.info(f"Scaler saved to {scaler_path}")
    
    except Exception as e:
        logging.error(f"Error in saving model: {e}")
        raise

def test_prediction_function():
    """
    Test the prediction function with sample CBC values.
    """
    try:
        # Load model and scaler
        model = joblib.load('anemia_model.joblib')
        scaler = joblib.load('anemia_scaler.joblib')
        
        # Sample CBC values
        sample_values = {
            'HGB': 12.5, 
            'RBC': 4.8, 
            'HCT': 38.0, 
            'MCV': 85.0
        }
        
        # Make prediction
        result, probability = predict_anemia(sample_values, model, scaler)
        logging.info(f"\nTest prediction for {sample_values}:")
        logging.info(f"Prediction: {result}")
        logging.info(f"Probability: {probability:.4f}")
    
    except Exception as e:
        logging.error(f"Error in testing prediction function: {e}")
        raise

def main():
    """
    Main function to run the anemia detection model pipeline.
    """
    try:
        # Set the file path
        file_path = '20250428_074907.csv'
        
        # Load and preprocess data
        logging.info("Starting anemia detection model pipeline...")
        X, y, feature_names = load_and_preprocess_data(file_path)
        
        # Visualize class distribution
        plot_class_distribution(y)
        
        # Train and evaluate models
        best_model, scaler = train_and_evaluate_models(X, y, feature_names)
        
        # Save model and scaler
        save_model(best_model, scaler)
        
        # Test prediction function
        test_prediction_function()
        
        logging.info("Anemia detection model pipeline completed successfully!")
    
    except Exception as e:
        logging.error(f"Error in anemia detection pipeline: {e}")
        raise

if __name__ == "__main__":
    main()
