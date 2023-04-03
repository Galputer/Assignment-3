import os
import pandas as pd
import numpy as np
# we don't need iris datasets, we have our own
#from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#starting file for machine learning new permissable heuristic
def ml(args):
    directory = 'board-data'
    testData = pd.DataFrame()
    
    # Load the a different dataset based on our boardsS
    ## we need a panda dataset, input from the Main.py script
    # iterate over files in
    # that directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            df = pd.read_csv(f)
            boardnums = df.to_numpy()
            ace_calc = boardnums[len(boardnums)-1]
            boardnums = np.delete(boardnums, -1)
            tempDF = pd.DataFrame({'matrix': boardnums, 'astar_calc':ace_calc})
            testData.append(tempDF, ignore_index=True)
        
    X, y = testData.data, testData.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a decision tree classifier and train it on the training set
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the accuracy of the classifier on the testing set
    accuracy = clf.score(X_test, y_test)
    print(f"Accuracy: {accuracy}")