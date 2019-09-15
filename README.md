# stock_analyze

Exersice prctice from [In 12 minutes: Stocks Analysis with Pandas and Scikit-Learn](https://towardsdatascience.com/in-12-minutes-stocks-analysis-with-pandas-and-scikit-learn-a8d8a7b50ee7)

Four different models result for predicting the stock in [StocksPriceModel.py](https://github.com/Jeff42code/stock_analyze/blob/master/StocksPriceModel.py)

Uncomment each result to show deffernt plot result
```
# Linear Regresseion result
#forecast_set = clfreg.predict(X_lately)

# Quadratic Regression 2 result
#forecast_set = clfpoly2.predict(X_lately)

# Quadratic Regression 3 result
#forecast_set = clfpoly3.predict(X_lately)

# KNN Regression result
forecast_set = clfknn.predict(X_lately)
```
