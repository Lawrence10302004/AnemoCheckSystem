#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Utilities for Anemia Detection
-----------------------------------
This module contains utility functions for the anemia detection model.
"""

import numpy as np
import pandas as pd
import joblib
import logging
import os

def load_model_and_scaler(model_path='anemia_model.joblib', scaler_path='anemia_scaler.joblib'):
    """
    Load the trained model and scaler from disk.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model file
    scaler_path : str
        Path to the saved scaler file
    
    Returns:
    --------
    model : sklearn estimator
        The loaded model
    scaler : sklearn.preprocessing.StandardScaler
        The loaded scaler
    """
    try:
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            
        # Load model and scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        return model, scaler
    
    except Exception as e:
        logging.error(f"Error loading model and scaler: {e}")
        raise

def predict_anemia(cbc_values, model=None, scaler=None):
    """
    Predict whether a patient has anemia based on CBC values.
    
    Parameters:
    -----------
    cbc_values : dict
        Dictionary containing CBC values with keys: 'HGB', 'RBC', 'HCT', 'MCV'
    model : sklearn estimator, optional
        Trained model for prediction (if None, will attempt to load from disk)
    scaler : sklearn.preprocessing.StandardScaler, optional
        Scaler for feature normalization (if None, will attempt to load from disk)
    
    Returns:
    --------
    result : str
        'Anemic' or 'Normal'
    probability : float
        Probability of anemia
    """
    try:
        # Load model and scaler if not provided
        if model is None or scaler is None:
            model, scaler = load_model_and_scaler()
        
        # Validate input
        required_features = ['HGB', 'RBC', 'HCT', 'MCV']
        for feature in required_features:
            if feature not in cbc_values:
                raise ValueError(f"Missing required feature: {feature}")
        
        # Create features dataframe in the correct order
        features = pd.DataFrame([
            [cbc_values['HGB'], cbc_values['RBC'], cbc_values['HCT'], cbc_values['MCV']]
        ], columns=required_features)
        
        # Scale features
        scaled_features = scaler.transform(features)
        
        # Make prediction
        probability = model.predict_proba(scaled_features)[0][1]
        prediction = model.predict(scaled_features)[0]
        
        # Return result
        result = 'Anemic' if prediction == 1 else 'Normal'
        
        return result, probability
    
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        raise

def get_normal_ranges():
    """
    Return the normal ranges for CBC parameters.
    
    Returns:
    --------
    dict
        Dictionary containing normal ranges for CBC parameters
    """
    return {
        'HGB': {
            'male': (13.5, 17.5),  # g/dL
            'female': (12.0, 15.5)  # g/dL
        },
        'RBC': {
            'male': (4.5, 5.9),  # million cells/mcL
            'female': (4.1, 5.1)  # million cells/mcL
        },
        'HCT': {
            'male': (41.0, 50.0),  # %
            'female': (36.0, 44.0)  # %
        },
        'MCV': {
            'universal': (80.0, 96.0)  # fL (femtoliters)
        }
    }

def get_anemia_types():
    """
    Return information about different types of anemia based on CBC parameters.
    
    Returns:
    --------
    dict
        Dictionary containing anemia types and their characteristics
    """
    return {
        'Iron deficiency anemia': {
            'HGB': 'Low',
            'MCV': 'Low (microcytic)',
            'characteristics': 'Most common type of anemia, caused by insufficient iron.'
        },
        'Vitamin B12 deficiency anemia': {
            'HGB': 'Low',
            'MCV': 'High (macrocytic)',
            'characteristics': 'Caused by lack of vitamin B12, essential for RBC production.'
        },
        'Folate deficiency anemia': {
            'HGB': 'Low',
            'MCV': 'High (macrocytic)',
            'characteristics': 'Caused by insufficient folate, needed for DNA synthesis.'
        },
        'Hemolytic anemia': {
            'HGB': 'Low',
            'MCV': 'Normal or high',
            'characteristics': 'Caused by destruction of RBCs faster than they can be made.'
        },
        'Aplastic anemia': {
            'HGB': 'Low',
            'RBC': 'Low',
            'characteristics': 'Caused by bone marrow failure to produce enough blood cells.'
        },
        'Thalassemia': {
            'HGB': 'Low',
            'MCV': 'Low (microcytic)',
            'characteristics': 'Genetic disorder affecting hemoglobin production.'
        }
    }
