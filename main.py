import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

# Constants
CUSTOMER_FILE = "teleCust.csv"

def load_and_prepare_data(file_path):
    """
    Load the dataset and perform basic exploratory data analysis.
    :param file_path: Path to the CSV file
    :return: Loaded DataFrame
    """
    df = pd.read_csv(file_path)
    
    # Display basic info about the data
    print("\nDataset Information:")
    print(df.info())
    
    # Display first few rows of the dataset
    print("\nFirst 5 Rows of the Dataset:")
    print(df.head())
    
    # Statistical summary of the dataset
    print("\nStatistical Summary:")
    print(df.describe())
    
    return df

def labelling_data(df):
    """
    Separate the target column 'custcat' from the feature set.
    :param df: DataFrame with the data
    :return: Feature matrix X and target vector y
    """
    y = df[['custcat']].values
    X = df.drop('custcat', axis=1).values  # Properly drop the 'custcat' column from X
    return X, y

def normalize_data(X):
    """
    Normalize the feature set using StandardScaler.
    :param X: Feature matrix
    :return: Normalized feature matrix
    """
    X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
    return X

def evaluate_model(y_test, y_pred, k):
    """
    Evaluate the model with accuracy, classification report, and confusion matrix.
    :param y_test: True labels
    :param y_pred: Predicted labels
    :param k: Number of neighbors used in KNN
    """
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy for k={k}: {acc:.2f}")
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix (k={k})")
    plt.ylabel("True Labels")
    plt.xlabel("Predicted Labels")
    plt.show()

    return acc

def find_best_k(X_train, X_test, y_train, y_test):
    """
    Iterate through different values of k to find the one with the best accuracy.
    :param X_train: Training features
    :param X_test: Test features
    :param y_train: Training labels
    :param y_test: Test labels
    :return: Best value of k and corresponding accuracy
    """
    best_k = 1
    best_accuracy = 0
    accuracies = []

    # Test different values of k
    for k in range(1, 21):
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(X_train, y_train)
        y_pred = neigh.predict(X_test)
        acc = evaluate_model(y_test, y_pred, k)
        accuracies.append(acc)
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_k = k
    
    # Plot accuracy vs k
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 21), accuracies, marker='o', linestyle='--', color='b')
    plt.title('Accuracy vs. k in KNN')
    plt.xlabel('Number of Neighbors k')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

    print(f"\nBest k: {best_k} with accuracy: {best_accuracy:.2f}")
    return best_k, best_accuracy

def main():
    # Load and prepare the dataset
    df = load_and_prepare_data(CUSTOMER_FILE)
    
    # Label the data and normalize it
    X, y = labelling_data(df)
    X = normalize_data(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Find the best k and plot accuracy vs k
    best_k, best_accuracy = find_best_k(X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()
