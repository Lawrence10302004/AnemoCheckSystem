#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick Anemia Detection Model Generator
-------------------------------------
This script quickly trains and saves a simple logistic regression model
for anemia detection using CBC data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to quickly generate and save an anemia detection model.
    """
    try:
        # Load the dataset
        logging.info("Loading dataset from 20250428_074907.csv")
        df = pd.read_csv('20250428_074907.csv')
        logging.info(f"Dataset shape: {df.shape}")
        
        # Convert categorical labels to numerical
        logging.info("Converting labels: 'Normal' -> 0, 'Anemia' -> 1")
        df['Label'] = df['Label'].map({'Normal': 0, 'Anemia': 1})
        
        # Select relevant features
        features = ['HGB', 'RBC', 'HCT', 'MCV']
        logging.info(f"Selected features: {features}")
        
        # Extract features and target
        X = df[features]
        y = df['Label']
        
        # Split the data (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logging.info(f"Data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train a simple logistic regression model
        logging.info("Training logistic regression model")
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        # Save model and scaler
        logging.info("Saving model and scaler")
        joblib.dump(model, 'anemia_model.joblib')
        joblib.dump(scaler, 'anemia_scaler.joblib')
        
        logging.info("Model and scaler saved successfully!")
        
    except Exception as e:
        logging.error(f"Error generating model: {e}")
        raise

if __name__ == "__main__":
    main()