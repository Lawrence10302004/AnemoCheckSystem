import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data
data = pd.read_csv('cbc_anemia_dataset.csv')
# data = data.drop('ID', axis=1)

# Encode categorical variables
# le_gender = LabelEncoder()
# data['Gender'] = le_gender.fit_transform(data['Gender'])

le_label = LabelEncoder()
data['Label'] = le_label.fit_transform(data['Label'])

# Features and target
X = data.drop('Label', axis=1)
y = data['Label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 3, 5, 10],
    'max_features': ['sqrt', 'log2', None],
    'min_samples_split': [2, 5, 10]
}

# Grid search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

# Fit model
grid_search.fit(X_train, y_train)

# Best model
best_clf = grid_search.best_estimator_

# Save model to joblib
joblib.dump(best_clf, 'best_rf_anemia_model.joblib')
print("Model saved as best_rf_anemia_model.joblib")

# Evaluate
y_pred = best_clf.predict(X_test)
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le_label.classes_))
