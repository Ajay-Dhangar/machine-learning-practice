import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def generate_random_variables():
    # Generate random variables from different distributions
    uniform_rv = np.random.uniform(0, 1, 1000)  # Uniform distribution
    normal_rv = np.random.normal(0, 1, 1000)    # Normal distribution
    exponential_rv = np.random.exponential(1, 1000)  # Exponential distribution
    
    return uniform_rv, normal_rv, exponential_rv

def plot_distributions(uniform_rv, normal_rv, exponential_rv):
    # Plot histograms of the random variables
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(uniform_rv, bins=30, density=True, alpha=0.6, color='g')
    plt.title('Uniform Distribution')
    
    plt.subplot(1, 3, 2)
    plt.hist(normal_rv, bins=30, density=True, alpha=0.6, color='b')
    plt.title('Normal Distribution')
    
    plt.subplot(1, 3, 3)
    plt.hist(exponential_rv, bins=30, density=True, alpha=0.6, color='r')
    plt.title('Exponential Distribution')
    
    plt.show()

def plot_probability_density_functions():
    # Define the range for the x-axis
    x = np.linspace(-3, 3, 1000)
    
    # Probability density functions
    uniform_pdf = stats.uniform.pdf(x, -1, 2)
    normal_pdf = stats.norm.pdf(x, 0, 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, uniform_pdf, 'g-', label='Uniform PDF')
    plt.title('Uniform Probability Density Function')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(x, normal_pdf, 'b-', label='Normal PDF')
    plt.title('Normal Probability Density Function')
    plt.legend()
    
    plt.show()

def compute_properties(normal_rv):
    # Compute mean, variance, skewness, and kurtosis of a normal distribution
    mean = np.mean(normal_rv)
    variance = np.var(normal_rv)
    skewness = stats.skew(normal_rv)
    kurtosis = stats.kurtosis(normal_rv)
    
    print(f"Mean: {mean}")
    print(f"Variance: {variance}")
    print(f"Skewness: {skewness}")
    print(f"Kurtosis: {kurtosis}")

def compute_probabilities():
    # Compute probabilities for a normal distribution
    mean = 0
    std_dev = 1
    probability_less_than_1 = stats.norm.cdf(1, mean, std_dev)
    probability_between_1_and_2 = stats.norm.cdf(2, mean, std_dev) - stats.norm.cdf(1, mean, std_dev)
    
    print(f"Probability of being less than 1: {probability_less_than_1}")
    print(f"Probability of being between 1 and 2: {probability_between_1_and_2}")

def main():
    # Generate random variables
    uniform_rv, normal_rv, exponential_rv = generate_random_variables()
    
    # Plot distributions
    plot_distributions(uniform_rv, normal_rv, exponential_rv)
    
    # Plot probability density functions
    plot_probability_density_functions()
    
    # Compute properties of the normal distribution
    compute_properties(normal_rv)
    
    # Compute probabilities for a normal distribution
    compute_probabilities()

if __name__ == "__main__":
    main()
