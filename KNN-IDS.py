import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load the dataset from a file. 

    Args:
        file_path (str): A path to the dataset.

    Returns:
        X (DataFrame): Features of the dataset.
        y (Series): Target variable of the dataset.
    """
    data = pd.read_csv(file_path, encoding='utf-8')  # specifying UTF-8 encoding
    X = data.drop('Label', axis=1)
    y = data['Label']
    return X, y

def prepare_data(X, y):
    """
    Split the data into training and testing sets.
    
    Args:
        X (array-like): The input features.
        y (array-like): The target variable.
        
    Returns:
        X_train (array-like): The training features.
        X_test (array-like): The testing features.
        y_train (array-like): The training target variable.
        y_test (array-like): The testing target variable.
    """
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train(X_train, y_train):
    """
    Train a k-nearest neighbors classifier model.

    Args:
        X_train (array-like): The training data features.
        y_train (array-like): The training data labels.

    Returns:
        KNeighborsClassifier: The trained classifier model.
    """
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    return knn

def evaluate(prediction, ground_truth):
    """
    Calculate the accuracy, precision, recall, and confusion matrix for a given prediction and ground truth.

    Args:
        prediction (array-like): The predicted values.
        ground_truth (array-like): The ground truth values.

    Returns:
        accuracy (float): The accuracy of the prediction.
        precision (float): The precision of the prediction.
        recall (float): The recall of the prediction.
        cm (array): The confusion matrix.
    """
    accuracy = accuracy_score(ground_truth, prediction)
    precision = precision_score(ground_truth, prediction)
    recall = recall_score(ground_truth, prediction)
    cm = confusion_matrix(ground_truth, prediction)
    return accuracy, precision, recall, cm

def main(): 
    X, y = load_data('network_packet_data.csv')

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = prepare_data(X, y)

    # Train the model
    knn = train(X_train, y_train)

    # Make predictions
    prediction = knn.predict(X_test)

    # Evaluate the model
    accuracy, precision, recall, cm = evaluate(prediction, y_test)

    # Print the results
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("Confusion matrix: ", cm)

if __name__ == "__main__":
    main()