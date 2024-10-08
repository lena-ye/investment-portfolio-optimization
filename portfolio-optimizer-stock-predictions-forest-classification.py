# Lena Ye
# October 2024
# Portfolio Optimization with Machine Learning
# This program is designed to predict stock price movements and optimize portfolio allocation based on historical financial data.

# Imports
import yfinance as yf # for downloading financial data
import pandas as pd # for data manipulation and analysis using dataframes
import numpy as np # for working with arrays and numerical operations
from sklearn.ensemble import RandomForestClassifier # for improving accuracy and reduce overfitting
from sklearn.model_selection import train_test_split # for training and testing set categorizations
from scipy.optimize import minimize # for finding the portfolio that gives the best return relative to volatility
import matplotlib.pyplot as plt # for displaying visual representations

# Fetching historical data for specific tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
# date selection: 2018 to 2023
data = yf.download(tickers, start="2018-01-01", end="2023-01-01")

# Sentiment score: reflects public opinion or overall attitude about a stock
# Here we randomize the sentiment score 
data['Sentiment'] = np.random.choice([1, -1], size=len(data))

# Engineering features: simple moving average (SMA) across 20 days and returns
for ticker in tickers:
    # Adds a new column with the SMA of each stock's adjusted closing price
    # Closing price is adjusted to reflect the stock's true value more accurately
    # by taking into account the dividends, stock splits, and other corp actions
    data[('SMA_20', ticker)] = data['Adj Close'][ticker].rolling(window=20).mean()
    # add a new column that calculates the percentage changes in the daily return of the adjusted close prices
    data[('Returns', ticker)] = data['Adj Close'][ticker].pct_change()

# Remove rows with missing data caused by rolling averages (20 days in this case)
data = data.dropna()

# Creating the dataframe containing features (inputs) for machine learning
# Using AAPL as an example for price movement prediction
features = pd.DataFrame({
    'SMA_20': data[('SMA_20', 'AAPL')], # 20 day SMA
    'Returns': data[('Returns', 'AAPL')], # daily returns
    'Sentiment': data['Sentiment'] # sentiment
})

# Define the target: 1 if AAPL's next day's price is higher and 0 otherwise
# boolean compares tomorrow's data with the ones from today
target = (data['Adj Close']['AAPL'].shift(-1) > data['Adj Close']['AAPL']).astype(int)

# Align features and target
# We select a subset of the features dataframe that corresponds to valid data in target series 
features = features.loc[target.dropna().index]
# drop missing values from target to ensure that no rows with incomplete data are left in the set
target = target.dropna()

# Split available data into training and testing sets
# X represents the input data (the features) and Y the target data
# We will use 70% of available data to train the model and 30% to test it
# we must set random_state to ensure reproducibility so the code goes through 
#   the same random process every time it is run
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)


# Train the Random Forest Classifier using 100 decision trees
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Check accuracy (proportion of correct predictions)
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")


# Portfolio optimization using Modern Portfolio Theory

# portfolio-performance is a function that calculate the annualized return and volatility of a portfolio
# weights: allocation of capital across different stocks, aka proportion of portfolio
#     dedicated to each stock
# mean-returns: average daily returns of each stock
# cov_matrix: covariance matrix of stock returns
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return returns, volatility

# Function to minimize the negative Sharpe ratio
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    returns, volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    sharpe_ratio = (returns - risk_free_rate) / volatility
    return -sharpe_ratio

# Get mean returns and covariance matrix
mean_returns = data['Adj Close'].pct_change().mean()
cov_matrix = data['Adj Close'].pct_change().cov()

# Constraints (weights sum to 1) and bounds (between 0 and 1)
# the result must be 0 for the constraint to be satisfied
constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
bounds = tuple((0, 1) for _ in range(len(tickers)))

# Assume even distribution of weights initially
initial_weights = np.array([1/len(tickers)] * len(tickers))

# Risk-free rate (here we use US treasury yield as a standard)
risk_free_rate = 0.01

# Minimize negative Sharpe ratio
# sequential least squares quadratic programming (SLSQP) is used to minimize risk and maximize returns
result = minimize(negative_sharpe_ratio, initial_weights, args=(mean_returns, cov_matrix, risk_free_rate),
                  method='SLSQP', bounds=bounds, constraints=constraints)

# Output optimal weights (proportion of capital allocated to each stock) to the user
optimal_weights = result.x
print(f"Optimal weights: {optimal_weights}")

# Plot the efficient frontier
def plot_efficient_frontier(mean_returns, cov_matrix, risk_free_rate):
    # we initialize a numpy array with three rows corresponding to returns, volatility, 
    #     and Sharpe ratio for 10000 simulated portfolios
    results = np.zeros((3, 10000))
    for i in range(10000):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        # calculate returns and volatility using portfolio_performance function
        returns, volatility = portfolio_performance(weights, mean_returns, cov_matrix)
        results[0, i] = returns
        results[1, i] = volatility
        results[2, i] = (returns - risk_free_rate) / volatility  # Sharpe ratio

    # We create specifications for the efficient frontier
    # the x axis is the Volatility
    # the y axis is the returns
    # the color of each datapoint corresponds to the Sharpe ratio of that portfolio
    plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', marker='o')
    plt.title('Efficient Frontier')
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('Returns')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()

# Output portfolio performance including returns and risks
opt_returns, opt_volatility = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
print(f"Optimal portfolio returns: {opt_returns:.2f}, volatility: {opt_volatility:.2f}")


# Plot the efficient frontier
# portfolios on the curve are best possible combinations of assets that provide
#     the highest return given a level of risk
# below the curve: inefficient as the offer a lower return for a given risk level
# above the curve: not possible as it requires achieving better returns with less risk
plot_efficient_frontier(mean_returns, cov_matrix, risk_free_rate)
