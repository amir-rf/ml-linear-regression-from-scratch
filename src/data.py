from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd

def load_california_housing():
    dataset = fetch_california_housing(as_frame=True)

    X = dataset.data
    y = dataset.target

    return X, y

def split_data(X, y, test_size=0.2, random_state=42,):
    return train_test_split(X, y,
                            test_size=test_size,
                            random_state=random_state)


if __name__ == "__main__":
    X, y = load_california_housing()

    X_train, X_test, y_train, y_test = split_data(X, y)
    print("Training features:", X_train.shape)
    print("Testing features:", X_test.shape)
    print("Training targets:", y_train.shape)
    print("Testing targets:", y_test.shape)
    print("\nFeature names:")
    print(list(X.columns))