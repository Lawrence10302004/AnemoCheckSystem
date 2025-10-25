#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib
# Use non-interactive backend to avoid opening GUI windows when the module is
# imported or run in a server environment.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os


def generate_synthetic_data(n_samples=1000, random_seed=42):
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
    # This creates a realistic spread of hemoglobin values
    hemoglobin = np.random.normal(loc=13, scale=2.5, size=n_samples)
    
    # Clip values to be within realistic range (4-18 g/dL)
    hemoglobin = np.clip(hemoglobin, 4, 18)
    
    # Create DataFrame
    X = pd.DataFrame({'hemoglobin': hemoglobin})
    
    # Classify according to WHO guidelines
    y = pd.Series(index=X.index, name='anemia_class', dtype='object')
    
    # Apply classification rules
    y[X['hemoglobin'] >= 12] = 'Normal'
    y[(X['hemoglobin'] >= 10) & (X['hemoglobin'] < 12)] = 'Mild'
    y[(X['hemoglobin'] >= 8) & (X['hemoglobin'] < 10)] = 'Moderate'
    y[X['hemoglobin'] < 8] = 'Severe'
    
    return X, y


def train_model(X, y):
    """
    Train a decision tree classifier with limited complexity for explainability.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Features (hemoglobin values)
    y : pandas.Series
        Anemia severity labels
    
    Returns:
    --------
    model : DecisionTreeClassifier
        Trained decision tree classifier
    X_train : pandas.DataFrame
        Training features
    X_test : pandas.DataFrame
        Testing features
    y_train : pandas.Series
        Training labels
    y_test : pandas.Series
        Testing labels
    """
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and train decision tree classifier
    # Max depth is limited to 3 for explainability
    model = DecisionTreeClassifier(
        max_depth=3,
        criterion='entropy',
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test data.
    
    Parameters:
    -----------
    model : DecisionTreeClassifier
        Trained decision tree classifier
    X_test : pandas.DataFrame
        Testing features
    y_test : pandas.Series
        Testing labels
    
    Returns:
    --------
    accuracy : float
        Model accuracy
    report : str
        Classification report
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return accuracy, report


def visualize_decision_tree(model, feature_names, class_names, save_path=None):
    """
    Visualize the decision tree with feature thresholds.
    
    Parameters:
    -----------
    model : DecisionTreeClassifier
        Trained decision tree classifier
    feature_names : list
        List of feature names
    class_names : list
        List of class names
    save_path : str, optional
        Path to save the visualization as PNG
    """
    plt.figure(figsize=(15, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        proportion=False,
        precision=2
    )
    plt.title("Decision Tree for Anemia Classification", fontsize=14)
    
    # Add WHO guidelines annotation
    guideline_text = (
        "WHO Anemia Guidelines:\n"
        "• Normal: Hemoglobin ≥ 12 g/dL\n"
        "• Mild: Hemoglobin 10-11.9 g/dL\n"
        "• Moderate: Hemoglobin 8-9.9 g/dL\n"
        "• Severe: Hemoglobin < 8 g/dL"
    )
    plt.annotate(
        guideline_text,
        xy=(0.05, 0.05),
        xycoords='figure fraction',
        bbox=dict(boxstyle="round,pad=0.5", fc="lightgray", alpha=0.8)
    )
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Decision tree visualization saved to {save_path}")

    plt.tight_layout()
    # Do not call plt.show() (interactive); instead just close the figure so
    # no GUI mainloop is required. This avoids Tcl/Tk errors when running under
    # web servers or background threads.
    plt.close()


def get_prediction_and_recommendation(model, hemoglobin_value):
    """
    Get prediction and medical recommendation based on hemoglobin value.
    
    Parameters:
    -----------
    model : DecisionTreeClassifier
        Trained decision tree classifier
    hemoglobin_value : float
        Hemoglobin value in g/dL
    
    Returns:
    --------
    result : dict
        Dictionary containing prediction and recommendation
    """
    # Medical recommendations based on anemia severity
    recommendations = {
        'Normal': "Maintain a healthy diet rich in iron, vitamin B12, and folate.",
        'Mild': "Consider dietary adjustments to increase iron intake and monitor hemoglobin "
               "levels in 1-2 months. Foods rich in iron include red meat, spinach, and legumes.",
        'Moderate': "Medical consultation recommended. Iron supplements may be prescribed. "
                   "Further testing might be needed to determine the underlying cause.",
        'Severe': "Emergency medical care required. Immediate consultation with a healthcare "
                 "provider is necessary as severe anemia can lead to serious complications."
    }
    
    # Make prediction
    try:
        # Reshape for single sample prediction
        X_pred = np.array([[hemoglobin_value]])
        predicted_class = model.predict(X_pred)[0]
        
        # Get probability estimates
        probabilities = model.predict_proba(X_pred)[0]
        class_names = model.classes_
        prob_dict = {class_name: prob for class_name, prob in zip(class_names, probabilities)}
        
        # Create result dictionary
        result = {
            'hemoglobin': hemoglobin_value,
            'predicted_class': predicted_class,
            'confidence': prob_dict[predicted_class],
            'recommendation': recommendations[predicted_class],
            'all_probabilities': prob_dict
        }
        
        return result
    
    except Exception as e:
        return {'error': f"Prediction error: {str(e)}"}


def print_prediction_result(result):
    """
    Print prediction result in a formatted way.
    
    Parameters:
    -----------
    result : dict
        Dictionary containing prediction and recommendation
    """
    if 'error' in result:
        print(f"\n❌ ERROR: {result['error']}")
        return
    
    # Set color based on anemia severity
    colors = {
        'Normal': '\033[92m',  # Green
        'Mild': '\033[93m',    # Yellow
        'Moderate': '\033[91m', # Red
        'Severe': '\033[91m\033[1m'  # Bold Red
    }
    
    # Reset color
    reset_color = '\033[0m'
    
    # Get color for current prediction
    color = colors.get(result['predicted_class'], '')
    
    print("\n" + "="*70)
    print(f"ANEMIA CLASSIFICATION RESULT")
    print("="*70)
    
    print(f"Hemoglobin Level: {result['hemoglobin']:.1f} g/dL")
    
    print(f"Classification: {color}{result['predicted_class']}{reset_color} "
          f"(Confidence: {result['confidence']*100:.1f}%)")
    
    print("\nProbabilities:")
    for cls, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
        cls_color = colors.get(cls, '')
        print(f"  {cls_color}{cls}{reset_color}: {prob*100:.1f}%")
    
    print("\nMedical Recommendation:")
    print(f"  {result['recommendation']}")
    
    print("="*70)


def user_interface(model):
    """
    Simple CLI interface for user interaction.
    
    Parameters:
    -----------
    model : DecisionTreeClassifier
        Trained decision tree classifier
    """
    print("\nANEMIA CLASSIFIER - WHO GUIDELINES")
    print("Enter 'q' to quit at any time\n")
    
    while True:
        # Get user input
        user_input = input("Enter hemoglobin level (g/dL): ")
        
        # Check if user wants to quit
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("\nThank you for using the Anemia Classifier. Goodbye!")
            break
        
        # Validate input
        try:
            hemoglobin = float(user_input)
            
            # Check for realistic range
            if hemoglobin < 1 or hemoglobin > 25:
                print("\n⚠️  Warning: The entered value is outside the typical range for hemoglobin "
                      "(1-25 g/dL). Please verify your measurement.")
                continue_anyway = input("Continue with this value anyway? (y/n): ")
                if continue_anyway.lower() != 'y':
                    continue
            
            # Get prediction and recommendation
            result = get_prediction_and_recommendation(model, hemoglobin)
            print_prediction_result(result)
            
        except ValueError:
            print("\n❌ ERROR: Please enter a valid number for hemoglobin level.")
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")


def main():
    """
    Main function to run the anemia classification program.
    """
    print("Anemia Classification using WHO Guidelines")
    print("=========================================")
    print("Generating synthetic dataset...")
    X, y = generate_synthetic_data(n_samples=1000)
    
    # Display dataset information
    print(f"Generated {len(X)} samples")
    print("\nClass Distribution:")
    for class_name, count in y.value_counts().items():
        print(f"  {class_name}: {count} samples ({count/len(y)*100:.1f}%)")
    
    print("\nTraining decision tree classifier...")
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    
    # Evaluate model
    accuracy, report = evaluate_model(model, X_test, y_test)
    print(f"\nModel accuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:")
    print(report)
    
    # Ask user if they want to see decision tree visualization
    show_viz = input("\nVisualize decision tree? (y/n): ")
    if show_viz.lower() == 'y':
        save_viz = input("Save visualization as PNG? (y/n): ")
        save_path = None
        if save_viz.lower() == 'y':
            save_path = "anemia_decision_tree.png"
        
        visualize_decision_tree(
            model, 
            feature_names=['Hemoglobin (g/dL)'], 
            class_names=sorted(y.unique()),
            save_path=save_path
        )
    
    # Start user interface
    user_interface(model)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram terminated by user. Goodbye!")
    except Exception as e:
        print(f"\n\nAn unexpected error occurred: {str(e)}")

"""
Future Enhancements (Commented Out):
-----------------------------------
The current implementation uses a Decision Tree classifier for simplicity and explainability.
For improved accuracy, consider using more sophisticated models like Random Forest or XGBoost.

# Random Forest Implementation
from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

# XGBoost Implementation
# Requires: pip install xgboost
# import xgboost as xgb

# def train_xgboost(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     
#     # Convert categorical to numerical
#     label_map = {label: i for i, label in enumerate(sorted(y.unique()))}
#     y_train_num = y_train.map(label_map)
#     y_test_num = y_test.map(label_map)
#     
#     model = xgb.XGBClassifier(
#         max_depth=3,
#         learning_rate=0.1,
#         n_estimators=100,
#         objective='multi:softprob',
#         num_class=len(label_map),
#         random_state=42
#     )
#     model.fit(X_train, y_train_num)
#     
#     # Add mapping back to the model for interpretation
#     model.classes_ = np.array(sorted(label_map.keys()))
#     
#     return model, X_train, X_test, y_train, y_test
"""