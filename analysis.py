import pandas as pd
import matplotlib.pyplot as plt

def load_csv(file_path):
    df = pd.read_csv(file_path)
    print("DataFrame Loaded:")
    print(df.head())
    return df

def plot_histograms(df):
    df.hist(figsize=(10, 6))
    plt.suptitle('Histograms')
    plt.show()

def plot_scatter(df, x_column, y_column):
    df.plot(kind='scatter', x=x_column, y=y_column, figsize=(10, 6), title=f'Scatter plot of {x_column} vs {y_column}')
    plt.show()

def plot_boxplot(df):
    df.plot(kind='box', figsize=(10, 6), title='Box Plots')
    plt.show()

def summary_statistics(df):
    print("Summary Statistics:")
    print(df.describe())

def get_column(df, column_name):
    return df[column_name]

def get_row(df, index):
    return df.iloc[index]

def filter_rows(df, condition):
    return df.query(condition)

def main():
    df = load_csv('data/data.csv')
    
    plot_histograms(df)
    plot_scatter(df, 'value1', 'value2')
    plot_boxplot(df)  

    summary_statistics(df)
    
    print("Accessing a Column:")
    print(get_column(df, 'value1'))
    
    print("Accessing a Row:")
    print(get_row(df, 1))
    
    print("Filtering Rows where value1 > 10:")
    print(filter_rows(df, 'value1 > 10'))

if __name__ == "__main__":
    main()