from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree
from feature_engineer import feature_engineer
from parse_board import parse_board

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import numpy as np

if __name__ == "__main__":
    board_df = parse_board()
    engineered_df = feature_engineer(board_df)
    # print(engineered_df.loc[:10].to_string(index=False))
    
    # Select the relevant features and target variable
    X = engineered_df
    y = board_df['astar']
    # print(X.loc[:10].to_string(index=False))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create a Lasso regression model
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

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
    
    #draw tree graph
    # Visualize the first decision tree in the Random Forest
    plt.figure(figsize=(20, 10))
    plot_tree(model.estimators_[0], feature_names=X.columns, filled=True)
    plt.show()

    # Visualize the feature importances of the Random Forest
    plt.figure(figsize=(10, 6))
    plt.bar(X.columns, model.feature_importances_)
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.show()