from feature_engineer import feature_engineer
from parse_board import parse_board

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

import numpy as np

if __name__ == "__main__":
    board_df = parse_board()
    engineered_df = feature_engineer(board_df)
    
    # Select the relevant features and target variable
    X = engineered_df
    y = board_df['astar']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create a Lasso regression model
    model = Lasso(alpha=0.1)

    # Fit the model to the data
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # y_pred = [int(i) for i in y_pred]
    
    # evaluate model
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    
    print("Train R^2: {:.2f}".format(train_r2))
    print("Test R^2: {:.2f}".format(test_r2))
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Test RMSE: {:.2f}".format(rmse))