import os
import pickle
from statistics import mean, median
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree
from feature_engineer import feature_engineer
from parse_board import parse_board

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate

import numpy as np

if __name__ == "__main__":
    print(f'{"Reading board data" :=<100}')
    board_df = parse_board()
    print(f'{"Generating features" :=<100}')
    engineered_df = feature_engineer(board_df)
    
    print(f'{"Running random forest" :=<100}')
    # Select the relevant features and target variable
    X = engineered_df.values
    y = board_df['astar'].values
    
    # Create a Lasso regression model
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    
    # Fit and evaluate the model using 10-fold cross-validation
    scores = cross_validate(model, X, y, cv=10, scoring=["r2","neg_root_mean_squared_error"])
    print(f'{"Generating random forest regressor statistics" :=<100}')
    # print("Cross-validation scores:", scores)
    print("Cross-validation mean RMSE: {:.2f}".format(0- mean(scores['test_neg_root_mean_squared_error'])))
    print("Cross-validation median RMSE: {:.2f}".format(0- median(scores['test_neg_root_mean_squared_error'])))

    print("Cross-validation mean R-squared: {:.2f}".format(mean(scores['test_r2'])))
    print("Cross-validation median R-squared: {:.2f}".format(median(scores['test_r2'])))

    # Calculate RMSE for the entire dataset
    model.fit(X, y)
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print("Model RMSE: {:.2f}".format(rmse))
    print("Model R-squared: {:.2f}".format(r2_score(y, y_pred)))
    
    print(f'{"Generating random forest regressor graphs" :=<100}')
    
    # Save the model as a pickle file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = 'models/random-forest-3.pkl'
    if os.path.isfile(os.path.join(dir_path,filename)):
        print(f'{"Model exists!" :=<100}')
    else:
        print(f'{"Saving model to disk" :=<100}')
        with open(os.path.join(dir_path,filename), 'wb') as file:
            pickle.dump(model, file)
    
    #draw tree graph
    # Visualize the first decision tree in the Random Forest
    plt.figure(figsize=(20, 10))
    plot_tree(model.estimators_[0], feature_names=engineered_df.columns, filled=True)
    plt.show()

    # Visualize the feature importances of the Random Forest
    plt.figure(figsize=(10, 6))
    plt.bar(engineered_df.columns, model.feature_importances_)
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.show()