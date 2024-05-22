import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/data.txt', delimiter=' ', skiprows=0)
print(df)  

plt.figure(figsize=(10, 6))
plt.bar(df['category'], df['value'])
plt.xlabel('Category')
plt.ylabel('Value')
plt.title('Bar Plot of Categories vs Values')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()