import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

df = pd.read_csv('weather.csv')

print(df.head())
print(df.info())
print(df.describe())

num_imputer = SimpleImputer(strategy='mean')
df[['Sunshine', 'WindGustSpeed', 'WindSpeed9am']] = num_imputer.fit_transform(df[['Sunshine', 'WindGustSpeed', 'WindSpeed9am']])

cat_imputer = SimpleImputer(strategy='most_frequent')
df[['WindGustDir', 'WindDir9am', 'WindDir3pm']] = cat_imputer.fit_transform(df[['WindGustDir', 'WindDir9am', 'WindDir3pm']])

df = pd.get_dummies(df, columns=['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow'], drop_first=True)

df['MaxTemp_squared'] = df['MaxTemp'] ** 2

plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

sns.pairplot(df[['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'Humidity9am', 'Humidity3pm']])
plt.show()

features = ['MinTemp', 'MaxTemp', 'MaxTemp_squared', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'Humidity9am', 'Humidity3pm']
X = df[features]
y = df['Rainfall']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred_lin = lin_reg.predict(X_test)

mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)
print(f'Linear Regression - Mean Squared Error: {mse_lin}')
print(f'Linear Regression - R-squared: {r2_lin}')

rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_train, y_train)

y_pred_rf = rf_reg.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f'Random Forest - Mean Squared Error: {mse_rf}')
print(f'Random Forest - R-squared: {r2_rf}')

cv_scores_rf = cross_val_score(rf_reg, X_scaled, y, cv=5, scoring='r2')
print(f'Random Forest - Cross-Validated R-squared: {cv_scores_rf.mean()}')

feature_importances = pd.Series(rf_reg.feature_importances_, index=features).sort_values(ascending=False)
print("Feature Importances from Random Forest:")
print(feature_importances)

plt.figure(figsize=(10, 6))
feature_importances.plot(kind='bar', color='forestgreen')
plt.title('Feature Importance from Random Forest')
plt.show()

residuals_rf = y_test - y_pred_rf
plt.figure(figsize=(10, 6))
sns.histplot(residuals_rf, kde=True, color='orange')
plt.title('Residuals Distribution - Random Forest')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.6, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2)
plt.title('Actual vs Predicted Rainfall - Random Forest')
plt.xlabel('Actual Rainfall')
plt.ylabel('Predicted Rainfall')
plt.show()

df.to_csv('weather_analysis_results.csv', index=False)

