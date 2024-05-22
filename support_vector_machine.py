import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_data():
    """Load the Iris dataset"""
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df

def explore_data(df):
    """Perform exploratory data analysis"""
    print("First few rows of the dataset:")
    print(df.head())
    print("\nSummary statistics:")
    print(df.describe())
    
    # Pairplot to visualize the relationships
    sns.pairplot(df, hue='target', markers=["o", "s", "D"])
    plt.show()

def preprocess_data(df):
    """Preprocess the data for training"""
    X = df.drop('target', axis=1)  # Features
    y = df['target']  # Target variable
    return X, y

def train_model(X_train, y_train):
    """Train the Support Vector Classifier"""
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    return svm

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance"""
    y_pred = model.predict(X_test)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nAccuracy Score:")
    print(accuracy_score(y_test, y_pred))

def main():
    # Load the dataset
    df = load_data()
    
    # Explore the data
    explore_data(df)
    
    # Preprocess the data
    X, y = preprocess_data(df)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the Support Vector Classifier
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
