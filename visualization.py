#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization Utilities for Anemia Detection
-------------------------------------------
This module contains visualization functions for the anemia detection model.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import logging

# Set style for all plots
plt.style.use('dark_background')
sns.set(font_scale=1.2)

def plot_class_distribution(y):
    """
    Plot the distribution of target classes.
    
    Parameters:
    -----------
    y : pandas.Series
        Target variable (0 for Normal, 1 for Anemia)
    """
    try:
        plt.figure(figsize=(10, 6))
        class_counts = y.value_counts()
        class_names = ['Normal', 'Anemia']
        ax = sns.barplot(x=[class_names[i] for i in class_counts.index], y=class_counts.values, palette='coolwarm')
        
        # Add counts and percentages on top of bars
        total = len(y)
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            percentage = 100 * height / total
            ax.text(p.get_x() + p.get_width()/2., height + 0.1,
                    f'{int(height)} ({percentage:.1f}%)',
                    ha="center", fontsize=12)
        
        plt.title('Class Distribution in Anemia Dataset', fontsize=16)
        plt.xlabel('Class', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.tight_layout()
        plt.savefig('class_distribution.png')
        plt.close()
        logging.info("Class distribution plot saved as 'class_distribution.png'")
    
    except Exception as e:
        logging.error(f"Error in plotting class distribution: {e}")
        plt.close()

def plot_confusion_matrix(cm, model_name):
    """
    Plot confusion matrix for model evaluation.
    
    Parameters:
    -----------
    cm : numpy.ndarray
        Confusion matrix
    model_name : str
        Name of the model
    """
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Anemia'],
                    yticklabels=['Normal', 'Anemia'])
        
        plt.title(f'Confusion Matrix - {model_name}', fontsize=16)
        plt.xlabel('Predicted label', fontsize=14)
        plt.ylabel('True label', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
        plt.close()
        logging.info(f"Confusion matrix plot saved as 'confusion_matrix_{model_name.replace(' ', '_').lower()}.png'")
    
    except Exception as e:
        logging.error(f"Error in plotting confusion matrix: {e}")
        plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, model_name):
    """
    Plot ROC curve for model evaluation.
    
    Parameters:
    -----------
    fpr : numpy.ndarray
        False positive rates
    tpr : numpy.ndarray
        True positive rates
    roc_auc : float
        Area under the ROC curve
    model_name : str
        Name of the model
    """
    try:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (area = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title(f'ROC Curve - {model_name}', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.tight_layout()
        plt.savefig(f'roc_curve_{model_name.replace(" ", "_").lower()}.png')
        plt.close()
        logging.info(f"ROC curve plot saved as 'roc_curve_{model_name.replace(' ', '_').lower()}.png'")
    
    except Exception as e:
        logging.error(f"Error in plotting ROC curve: {e}")
        plt.close()

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance for tree-based models.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model with feature_importances_ attribute
    feature_names : list
        Names of the features
    """
    try:
        if not hasattr(model, 'feature_importances_'):
            logging.warning("Model does not have feature_importances_ attribute. Skipping feature importance plot.")
            return
            
        # Get feature importance
        feature_importance = model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(feature_importance)[::-1]
        sorted_feature_names = [feature_names[i] for i in indices]
        sorted_importance = feature_importance[indices]
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x=sorted_importance, y=sorted_feature_names, palette='viridis')
        plt.title('Feature Importance for Anemia Detection', fontsize=16)
        plt.xlabel('Importance', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        logging.info("Feature importance plot saved as 'feature_importance.png'")
    
    except Exception as e:
        logging.error(f"Error in plotting feature importance: {e}")
        plt.close()

def plot_correlation_matrix(X):
    """
    Plot correlation matrix of features.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Features dataframe
    """
    try:
        plt.figure(figsize=(10, 8))
        corr = X.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, 
                    fmt=".2f", linewidths=0.5, square=True, center=0)
        plt.title('Correlation Matrix of CBC Features', fontsize=16)
        plt.tight_layout()
        plt.savefig('correlation_matrix.png')
        plt.close()
        logging.info("Correlation matrix plot saved as 'correlation_matrix.png'")
    
    except Exception as e:
        logging.error(f"Error in plotting correlation matrix: {e}")
        plt.close()

def plot_feature_distributions(X, y):
    """
    Plot distribution of each feature by target class.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Features dataframe
    y : pandas.Series
        Target variable (0 for Normal, 1 for Anemia)
    """
    try:
        # Combine features and target
        data = X.copy()
        data['Label'] = y
        data['Label'] = data['Label'].map({0: 'Normal', 1: 'Anemia'})
        
        # Create subplot for each feature
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(X.columns):
            sns.kdeplot(data=data, x=feature, hue='Label', ax=axes[i], fill=True, common_norm=False, palette='coolwarm')
            axes[i].set_title(f'Distribution of {feature} by Class', fontsize=14)
            axes[i].set_xlabel(feature, fontsize=12)
            axes[i].set_ylabel('Density', fontsize=12)
            axes[i].legend(title='Class', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('feature_distributions.png')
        plt.close()
        logging.info("Feature distributions plot saved as 'feature_distributions.png'")
    
    except Exception as e:
        logging.error(f"Error in plotting feature distributions: {e}")
        plt.close()

def plot_learning_curve(train_sizes, train_scores, test_scores, model_name):
    """
    Plot learning curve for model evaluation.
    
    Parameters:
    -----------
    train_sizes : numpy.ndarray
        Training set sizes
    train_scores : numpy.ndarray
        Training scores for each training size
    test_scores : numpy.ndarray
        Test scores for each training size
    model_name : str
        Name of the model
    """
    try:
        plt.figure(figsize=(10, 6))
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
        
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
        
        plt.title(f'Learning Curve - {model_name}', fontsize=16)
        plt.xlabel('Training Set Size', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.legend(loc="best", fontsize=12)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'learning_curve_{model_name.replace(" ", "_").lower()}.png')
        plt.close()
        logging.info(f"Learning curve plot saved as 'learning_curve_{model_name.replace(' ', '_').lower()}.png'")
    
    except Exception as e:
        logging.error(f"Error in plotting learning curve: {e}")
        plt.close()
