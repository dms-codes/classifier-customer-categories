 
# K-Nearest Neighbors (KNN) Classification with Accuracy vs. K Plot

This project demonstrates how to implement the K-Nearest Neighbors (KNN) algorithm for classification on a customer dataset. The program iterates through different values of `k` (number of neighbors) and plots the accuracy against `k`. The goal is to identify the optimal number of neighbors that yield the highest accuracy.

## Table of Contents

- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)
- [Code Overview](#code-overview)
- [Results](#results)

## Dataset

The dataset used in this project is `teleCust.csv`, which contains customer data with various features. The target variable is `custcat`, which categorizes the customers into different segments.

### Columns in the dataset:
- **custcat**: The target class (customer category).
- **Other features**: Numerical or categorical features used for classification.

## Dependencies

Before running the code, make sure you have the following Python libraries installed:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

## How to Run

1. Clone the repository or download the code.
2. Ensure the `teleCust.csv` file is in the same directory as the script.
3. Run the script using Python:

```bash
python knn_classification.py
```

The script will:
- Load the dataset.
- Normalize the feature data.
- Split the data into training and test sets.
- Iterate over different values of `k` to train the KNN model.
- Plot the accuracy against `k` and display the best `k` with its corresponding accuracy.

## Code Overview

### Main Functions

- **`load_and_prepare_data(file_path)`**: 
    Loads and performs basic exploratory analysis on the dataset. It displays information about the dataset and a statistical summary.

- **`labelling_data(df)`**: 
    Separates the target column `custcat` from the feature matrix `X`.

- **`normalize_data(X)`**: 
    Normalizes the feature set using `StandardScaler`.

- **`evaluate_model(y_test, y_pred, k)`**: 
    Evaluates the KNN model by computing the accuracy score and plotting a confusion matrix.

- **`find_best_k(X_train, X_test, y_train, y_test)`**: 
    Iterates through values of `k` (from 1 to 20) to find the one that gives the highest accuracy. Plots accuracy against `k`.

- **`main()`**: 
    This is the main function that ties everything together. It loads the dataset, prepares the features, trains the KNN model for different values of `k`, and plots the accuracy results.

### Example Plot

The script will generate a plot similar to the one below, showing the accuracy vs. different values of `k`.

![Accuracy vs k](https://via.placeholder.com/600x300?text=Accuracy+vs+k+Plot)

## Results

After running the script, you will see the optimal value of `k` (number of neighbors) that results in the highest accuracy for the test set.

The output will display:
- Accuracy for each value of `k`.
- The optimal `k` and its accuracy.
- A confusion matrix for each `k`.

Example output:

```
Accuracy for k=1: 0.82
Accuracy for k=2: 0.80
Accuracy for k=3: 0.85
...
Best k: 5 with accuracy: 0.86
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
 
