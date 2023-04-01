# we don't need iris datasets, we have our own
#from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#starting file for machine learning new permissable heuristic
def ml(args):

    # Load the a different dataset based on our boardsS
    ## we need a panda dataset, input from the Main.py script
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a decision tree classifier and train it on the training set
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the accuracy of the classifier on the testing set
    accuracy = clf.score(X_test, y_test)
    print(f"Accuracy: {accuracy}")