# src/model_trainer.py

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def train_logistic_regression(X_train, y_train):
    """
    Trains a logistic regression model on scaled training data.
    
    Returns:
        model: Trained LogisticRegression model
        scaler: Fitted StandardScaler used for scaling the training data
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_train_scaled, y_train)

    return model, scaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def train_random_forest_with_smote(X, y, random_state=42):
    """
    Train a Random Forest classifier after applying SMOTE on training data.

    Parameters:
    - X: Features (DataFrame or ndarray)
    - y: Target labels
    - random_state: Seed for reproducibility

    Returns:
    - model: Trained RandomForestClassifier
    - scaler: Fitted StandardScaler
    """
    # Split into training and test sets (you can skip this if already split outside)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply SMOTE
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_resampled, y_resampled)

    return model, scaler