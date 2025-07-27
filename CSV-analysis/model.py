import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV
df = pd.read_csv('students_scores.csv')

# Display basic stats
print("\nDataset Info:")
print(df.info())
print("\nDescriptive Stats:")
print(df.describe())

# Calculate average of math score
avg_math = df['math score'].mean()
print(f"\nAverage Math Score: {avg_math:.2f}")

# Bar chart: average scores by gender
df.groupby('gender')[['math score', 'reading score', 'writing score']].mean().plot(kind='bar')
plt.title("Average Scores by Gender")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Scatter plot: reading vs writing
plt.scatter(df['reading score'], df['writing score'], alpha=0.6)
plt.title("Reading Score vs Writing Score")
plt.xlabel("Reading Score")
plt.ylabel("Writing Score")
plt.grid(True)
plt.tight_layout()
plt.show()

# Correlation heatmap
corr = df[['math score', 'reading score', 'writing score']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Score Correlation Heatmap")
plt.tight_layout()
plt.show()

# ✅ Insights
print("\nInsights:")
print("1. Students who score well in reading often score well in writing.")
print("2. Female students tend to have slightly higher average scores across subjects.")
