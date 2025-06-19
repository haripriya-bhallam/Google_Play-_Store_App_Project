# Google_Play-_Store_App_Project
# Step 1: Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

#Step 2: Load Dataset
df = pd.read_csv("googleplaystore.csv")
print("Original Dataset Shape:", df.shape)

# Step 3: Data Cleaning
df.drop_duplicates(inplace=True)
df = df[df['Rating'].notna() & (df['Rating'] <= 5)]
df = df[df['App'].notna() & df['Category'].notna()]

# Convert columns to proper types
df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
df['Installs'] = df['Installs'].str.replace(r'[+,]', '', regex=True)
df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')
df['Price'] = df['Price'].str.replace('$', '', regex=False)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Handle 'Size' column
def parse_size(val):
    try:
        if 'M' in val:
            return float(val.replace('M', '')) * 1_000_000
        elif 'k' in val:
            return float(val.replace('k', '')) * 1_000
        else:
            return np.nan
    except:
        return np.nan

df['Size_Bytes'] = df['Size'].apply(parse_size)
df['Size_MB'] = df['Size_Bytes'] / 1_000_000

# Step 4: Feature Engineering
df['Free_Flag'] = df['Price'] == 0
df['Install_Group'] = pd.cut(df['Installs'], 
                             bins=[-1, 100, 10_000, 1_000_000, 100_000_000, np.inf],
                             labels=['Tiny', 'Small', 'Medium', 'Large', 'Huge'])

# Impute missing ratings by install group mean
df['Rating'] = df.groupby('Install_Group')['Rating'].transform(lambda x: x.fillna(x.mean()))

# Final cleanup
df.dropna(inplace=True)

# Step 5: Exploratory Visualizations
plt.figure(figsize=(8, 5))
sns.histplot(df['Rating'], bins=30, kde=True)
plt.title("App Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Category', y='Rating')
plt.xticks(rotation=90)
plt.title("Ratings per App Category")
plt.tight_layout()
plt.show()

# Step 6: Machine Learning - Predicting Rating
le = LabelEncoder()
df['Category_Code'] = le.fit_transform(df['Category'])

features = ['Reviews', 'Size_MB', 'Price', 'Category_Code']
X = df[features]
y = df['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\n RMSE of Linear Regression Model: {rmse:.4f}")

# Step 7: Correlation Insights
corr_matrix = df[['Rating', 'Reviews', 'Installs', 'Size_MB', 'Price']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="viridis", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# Pearson Correlation
corr_reviews_installs, _ = pearsonr(df['Reviews'], df['Installs'])
print(f"\n Pearson Correlation between Reviews and Installs: {corr_reviews_installs:.4f}")

#Step 8: SQL Export
conn = sqlite3.connect("original_play_store.db")
df.to_sql("apps", conn, if_exists="replace", index=False)

query = """
SELECT Category, COUNT(*) as Total_Apps, ROUND(AVG(Rating), 2) as Avg_Rating
FROM apps
GROUP BY Category
ORDER BY Avg_Rating DESC
LIMIT 10;
"""
top_categories = pd.read_sql(query, conn)
print("\n Top Categories by Average Rating:")
print(top_categories)

#Step 9: Excel Export
df.to_excel("Original_Google_Play_Store_Data.xlsx", index=False)
print("\n Data exported to Excel and SQLite successfully.")
