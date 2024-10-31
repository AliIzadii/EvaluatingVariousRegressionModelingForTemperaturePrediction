import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, QuantileRegressor
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from math import sqrt
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('GlobalWeatherRepository.csv')
df = pd.DataFrame(data)
df = df[df['temperature_celsius'] > 0]

numeric_df = df.select_dtypes(include=[np.number])
features = list(map(lambda x: x[0],
                        list(filter(lambda x: x[1]>0.1, 
                                    list(sorted({key: abs(value) for key, value in dict(numeric_df.corr()['temperature_celsius']).items()
                                                }.items(), 
                                                key=lambda item: item[1]
                                               )
                                        )
                                   )
                            )
                       )  
                   )
features = [feature for feature in features if feature != 'temperature_celsius']

y = df['temperature_celsius'].values.reshape(-1, 1)
x = df[features]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = sqrt(mse)
    r2 = r2_score(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    return predictions, mae, mse, rmse, r2, mape

results = {}

# 1. Linear Regression
lr = LinearRegression()
lr.fit(x_train_scaled, y_train)
results['Linear Regression'] = evaluate_model(lr, x_test_scaled, y_test)

# 2. Quantile Regression
qr = QuantileRegressor()
qr.fit(x_train_scaled, y_train.ravel())
results['Quantile Regression'] = evaluate_model(qr, x_test_scaled, y_test)

# 3. Ridge Regression
ridge = Ridge()
ridge.fit(x_train_scaled, y_train)
results['Ridge Regression'] = evaluate_model(ridge, x_test_scaled, y_test)

# 4. Lasso Regression
lasso = Lasso()
lasso.fit(x_train_scaled, y_train)
results['Lasso Regression'] = evaluate_model(lasso, x_test_scaled, y_test)

# 5. Elastic Net Regression
en = ElasticNet()
en.fit(x_train_scaled, y_train)
results['Elastic Net Regression'] = evaluate_model(en, x_test_scaled, y_test)

# 6. Principal Component Regression (using PCA + Linear Regression)
pca = PCA(n_components=min(x_train.shape[1], 10))  # Choose optimal components
x_train_pca = pca.fit_transform(x_train_scaled)
x_test_pca = pca.transform(x_test_scaled)
pcr = LinearRegression()
pcr.fit(x_train_pca, y_train)
results['Principal Component Regression'] = evaluate_model(pcr, x_test_pca, y_test)

# 7. Partial Least Squares Regression
pls = PLSRegression(n_components=min(x_train.shape[1], 10))
pls.fit(x_train_scaled, y_train)
results['Partial Least Squares Regression'] = evaluate_model(pls, x_test_scaled, y_test)

# 8. Support Vector Regression
svr = SVR()
svr.fit(x_train_scaled, y_train.ravel())
results['Support Vector Regression'] = evaluate_model(svr, x_test_scaled, y_test)

for model_name, metrics in results.items():
    print(f"{model_name} Results:")
    print(f"MAE: {metrics[1]}")
    print(f"MSE: {metrics[2]}")
    print(f"RMSE: {metrics[3]}")
    print(f"R2: {metrics[4]}")
    print(f"MAPE: {metrics[5]}\n")

plt.figure(figsize=(20, 16))

for i, (model_name, metrics) in enumerate(results.items()):
    predictions = metrics[0]
    actual = y_test.flatten()
    plt.subplot(4, 2, i + 1)
    plt.scatter(actual, predictions, color='blue', alpha=0.6, label='Predictions')
    plt.scatter(actual, actual, color='red', alpha=0.3, label='Actual Values')
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], color='black', linestyle='--', lw=2)
    plt.title(f'{model_name}: Actual vs. Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()

plt.tight_layout()
plt.show()

numerical_data = data.select_dtypes(include=['float64', 'int64'])

corr_matrix = numerical_data.corr()

plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix for Weather Data Features')
plt.show()