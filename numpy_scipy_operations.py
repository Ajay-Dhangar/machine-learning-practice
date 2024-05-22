import numpy as np
from scipy import linalg, optimize, integrate
import matplotlib.pyplot as plt

# Basic NumPy Operations

def numpy_operations():
    # Array creation
    array_1d = np.array([1, 2, 3, 4, 5])
    array_2d = np.array([[1, 2], [3, 4], [5, 6]])
    
    print("1D Array:", array_1d)
    print("2D Array:\n", array_2d)
    
    # Basic arithmetic operations
    array_sum = array_1d + 2
    array_multiplication = array_1d * 2
    array_elementwise_multiplication = array_1d * array_1d
    
    print("Array Sum:", array_sum)
    print("Array Multiplication:", array_multiplication)
    print("Elementwise Multiplication:", array_elementwise_multiplication)
    
    # Linear Algebra
    matrix = np.array([[1, 2], [3, 4]])
    vector = np.array([5, 6])
    
    matrix_inverse = linalg.inv(matrix)
    matrix_det = linalg.det(matrix)
    matrix_eigvals = linalg.eigvals(matrix)
    matrix_vector_product = np.dot(matrix, vector)
    
    print("Matrix:\n", matrix)
    print("Matrix Inverse:\n", matrix_inverse)
    print("Matrix Determinant:", matrix_det)
    print("Matrix Eigenvalues:", matrix_eigvals)
    print("Matrix-Vector Product:", matrix_vector_product)
    
    # Statistics
    mean = np.mean(array_1d)
    median = np.median(array_1d)
    std_dev = np.std(array_1d)
    
    print("Mean:", mean)
    print("Median:", median)
    print("Standard Deviation:", std_dev)

# Basic SciPy Operations

def scipy_operations():
    # Optimization
    def objective_function(x):
        return x**2 + 5*np.sin(x)
    
    result = optimize.minimize(objective_function, x0=0)
    print("Optimization Result:", result)
    
    # Integration
    def integrand(x):
        return np.exp(-x**2)
    
    integral_result, integral_error = integrate.quad(integrand, -np.inf, np.inf)
    print("Integral Result:", integral_result)
    print("Integral Error:", integral_error)
    
    # Example Plot
    x_values = np.linspace(-10, 10, 400)
    y_values = objective_function(x_values)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label='f(x) = x^2 + 5sin(x)')
    plt.scatter(result.x, result.fun, color='red', label='Minimum')
    plt.title('Objective Function Optimization')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.show()

def main():
    print("NumPy Operations:")
    numpy_operations()
    
    print("\nSciPy Operations:")
    scipy_operations()

if __name__ == "__main__":
    main()
