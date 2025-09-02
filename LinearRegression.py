import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv('Housing.csv')

print(df.head())

print(df.info())

# sns.pairplot(df)
# plt.show()


df_copy = df.copy()

yes_no_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for cols in yes_no_columns:
    df_copy[cols] = df_copy[cols].map({'yes': 1, 'no': 0})
    
df_copy = df_copy.drop(columns = ['furnishingstatus'])

x = df_copy[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea']]
y = df_copy['price']

plt.scatter(df_copy['area'], df_copy['price'])
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Area vs Price')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x=df_copy['bedrooms'], y=df_copy['price'])
plt.xlabel('Number of Bedrooms')
plt.ylabel('Price')
plt.title('Bedrooms vs Price')
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df_copy.corr(), annot=True, cmap='coolwarm', fmt=".3f")
plt.title('Correlation Matrix')
plt.show()


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions
y_predict = model.predict(X_test)
# Evaluate the model
sns.scatterplot(x=y_test, y=y_predict)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()

print(model.score(X_test, y_test))

