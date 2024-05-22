import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_data(file_path):
    """Load the dataset from a CSV file"""
    df = pd.read_csv(file_path)
    return df

def explore_data(df):
    """Perform exploratory data analysis"""
    print("First few rows of the dataset:")
    print(df.head())
    print("\nSummary statistics:")
    print(df.describe())
    
    # Visualize the relationship between features and target
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(df['Rooms'], df['Price'], alpha=0.6, color='b')
    plt.xlabel('Rooms')
    plt.ylabel('Price')
    plt.title('Rooms vs Price')
    
    plt.subplot(1, 2, 2)
    plt.scatter(df['Age'], df['Price'], alpha=0.6, color='r')
    plt.xlabel('Age')
    plt.ylabel('Price')
    plt.title('Age vs Price')
    
    plt.show()

def preprocess_data(df):
    """Preprocess the data for training"""
    X = df[['Rooms', 'Age']]  # Features
    y = df['Price']  # Target variable
    return X, y

def train_model(X_train, y_train):
    """Train the linear regression model"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance"""
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'R-squared (R2 Score): {r2}')
    
    # Plotting actual vs predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted Prices')
    plt.show()

def main():
    # Load the dataset
    df = load_data('data/simple_train.csv')
    
    # Explore the data
    explore_data(df)
    
    # Preprocess the data
    X, y = preprocess_data(df)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the linear regression model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
