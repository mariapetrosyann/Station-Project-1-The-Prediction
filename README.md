# Microsoft Stock Price Prediction

## Introduction
This project focuses on predicting the closing price of Microsoft (MSFT) stock using historical stock data. The dataset includes columns such as date, open, high, low, close, adjusted close, and volume. For this project, the 'Close' column is used as the target variable (y). The project involves data visualization, model selection, training, and evaluation using a Decision Tree Regression model.

## Getting Started

### Prerequisites
Ensure you have the following libraries installed:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install these libraries using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Station-Project-1-The-Prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd microsoft_stock_price_prediction.py
    ```

## Running the Project

### Running Locally
1. Ensure all required libraries are installed.
2. Download the MSFT dataset via this link (https://finance.yahoo.com/quote/MSFT/history/)
3. Open the project file (`microsoft_stock_price_prediction.py`) in your preferred IDE or text editor.
4. Run the script to execute the code, visualize data, and evaluate the model.

### Running on Google Colab
1. Open the provided Google Colab link in the main code (7th row 7th row in (`microsoft_stock_price_prediction.py`) project directory).
2. Run all cells to see the visualizations and evaluations.

## Libraries and Functions Used

- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical computations.
- **matplotlib**: For creating static, animated, and interactive visualizations.
- **seaborn**: For statistical data visualization.
- **scikit-learn**: For machine learning algorithms and model evaluation metrics.

## Project Steps and Outputs

### 1. Data Visualization
Visualized the relationships between different features using:
- Pair plots
- Heatmaps

Example output:
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Pair plot
sns.pairplot(df)
plt.show()

# Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
```

### 2. Data Splitting and Model Selection
Split the data into training and testing sets, then selected Decision Tree Regression for model training.

Example:
```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Splitting the data
X = df.drop(['Close'], axis=1)
y = df['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
```

### 3. Model Evaluation
Calculated Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²) metrics to evaluate the model's performance.

Example:
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Predictions
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")
```

## Results
The model's evaluation metrics are as follows:
- **MSE**: 4.480622
- **MAE**: 2.116748
- **RMSE**: 1.164735
- **R²**: 0.996896

These results demonstrate the accuracy and prediction capabilities of the Decision Tree Regression model for Microsoft stock price prediction.

## Conclusion
This project showcases the process of predicting stock prices using historical data and machine learning techniques. By following the steps provided, you can run and test the model, visualize the data, and evaluate the model's performance.
