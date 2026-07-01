
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("final_cleaned_dataset.csv")

features = df[
    ['Sleep Duration',
     'Quality of Sleep',
     'Physical Activity Level',
     'Stress Level']
]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

k = 3
print("\nNumber of clusters chosen:", k)
print("\nChosen features for clustering:", list(features.columns))

kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(scaled_features)
df['Cluster'] = kmeans.labels_


plt.figure(figsize=(8, 6))
plt.scatter(
    pca_features[:, 0],
    pca_features[:, 1],
    c=df['Cluster'],
    cmap='viridis'
)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("K-Means Clustering on Sleep Health Data (PCA Reduced)")
plt.colorbar(label="Cluster")
plt.show()

#----------------------------------------------------------------


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("final_cleaned_dataset.csv")
print("Dataset loaded")

numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

df['Steps_per_Hour_Sleep'] = df['Daily Steps'] / df['Sleep Duration']

df['Stress_Category'] = pd.cut(df['Stress Level'],
                               bins=[-1, 3, 6, 10],
                               labels=['Low', 'Medium', 'High'])

mean_sleep = np.mean(df['Sleep Duration'])
median_steps = np.median(df['Daily Steps'])
max_hr = np.max(df['Heart Rate'])
print(f"\nMean Sleep Duration: {mean_sleep:.2f} hours")
print(f"Median Daily Steps: {median_steps}")
print(f"Max Heart Rate: {max_hr}")

plt.figure(figsize=(8,5))
plt.hist(df['Sleep Duration'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Sleep Duration')
plt.xlabel('Hours of Sleep')
plt.ylabel('Number of People')
plt.show()

plt.figure(figsize=(8,5))
plt.hist(df['Daily Steps'], bins=15, color='lightgreen', edgecolor='black')
plt.title('Distribution of Daily Steps')
plt.xlabel('Daily Steps')
plt.ylabel('Number of People')
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(df['BMI Category'], df['Sleep Duration'], c=df['Heart Rate'], cmap='plasma', alpha=0.7)
plt.colorbar(label='Heart Rate')
plt.title('Sleep Duration vs BMI Category (colored by Heart Rate)')
plt.xlabel('BMI Category (0=Normal, 1=Overweight, 2=Obese)')
plt.ylabel('Sleep Duration (hours)')
plt.show()

if 'Sleep Disorder' in df.columns:
    disorder_counts = df['Sleep Disorder'].value_counts()
    plt.figure(figsize=(7,7))
    plt.pie(disorder_counts, labels=disorder_counts.index, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
    plt.title('Distribution of Sleep Disorder')
    plt.show()
else:
    print("Column 'Sleep Disorder' not found in dataset.")



#----------------------------------------------------------------------------------------


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("final_cleaned_dataset.csv")
print("Dataset loaded")
print(df.head())

X = df[['Daily Steps']]
y = df['Sleep Duration']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")
plt.figure(figsize=(8,5))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.title('Simple Linear Regression: Sleep Duration vs Daily Steps')
plt.xlabel('Daily Steps')
plt.ylabel('Sleep Duration (hours)')
plt.legend()
plt.show()


#-----------------------------------------------------------------------------------------


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("final_cleaned_dataset.csv")
print("Dataset loaded")

numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

df['Steps_per_Hour_Sleep'] = df['Daily Steps'] / df['Sleep Duration']

df['Stress_Category'] = pd.cut(df['Stress Level'],
                               bins=[-1, 3, 6, 10],
                               labels=['Low', 'Medium', 'High'])

mean_sleep = np.mean(df['Sleep Duration'])
median_steps = np.median(df['Daily Steps'])
max_hr = np.max(df['Heart Rate'])
print(f"\nMean Sleep Duration: {mean_sleep:.2f} hours")
print(f"Median Daily Steps: {median_steps}")
print(f"Max Heart Rate: {max_hr}")

plt.figure(figsize=(8,5))
plt.hist(df['Sleep Duration'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Sleep Duration')
plt.xlabel('Hours of Sleep')
plt.ylabel('Number of People')
plt.show()

plt.figure(figsize=(8,5))
plt.hist(df['Daily Steps'], bins=15, color='lightgreen', edgecolor='black')
plt.title('Distribution of Daily Steps')
plt.xlabel('Daily Steps')
plt.ylabel('Number of People')
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(df['BMI Category'], df['Sleep Duration'], c=df['Heart Rate'], cmap='plasma', alpha=0.7)
plt.colorbar(label='Heart Rate')
plt.title('Sleep Duration vs BMI Category (colored by Heart Rate)')
plt.xlabel('BMI Category (0=Normal, 1=Overweight, 2=Obese)')
plt.ylabel('Sleep Duration (hours)')
plt.show()

if 'Sleep Disorder' in df.columns:
    disorder_counts = df['Sleep Disorder'].value_counts()
    plt.figure(figsize=(7,7))
    plt.pie(disorder_counts, labels=disorder_counts.index, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
    plt.title('Distribution of Sleep Disorder')
    plt.show()
else:
    print("Column 'Sleep Disorder' not found in dataset.")
