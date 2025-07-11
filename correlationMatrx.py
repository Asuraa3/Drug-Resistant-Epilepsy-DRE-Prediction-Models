import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your features CSV
features = pd.read_csv('output/features.csv')

# Calculate the correlation matrix
corr_matrix = features.corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Feature Correlation Matrix")
plt.savefig('output/feature_correlation_matrix.png')
plt.show()


