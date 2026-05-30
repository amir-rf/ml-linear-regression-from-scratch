import numpy as np

from data import load_california_housing, split_data
from preprocessing import scale_features
from ridge_regression import RidgeRegressionScratch
from evaluate import compute_regression_metrics


def weight_l2_norm(weights):
    return np.linalg.norm(weights)


def main():

    X, y = load_california_housing()
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    y_train_array = y_train.to_numpy()
    y_test_array = y_test.to_numpy()

    alphas = [0, 0.01, 0.1, 1, 10, 100, 1000]

    print("\n"+"="*80)
    print("Ridge regression experiment")
    print("="*80)

    print(
        f"{'Alpha':>10} ",
        f"{'RMSE':>12} ",
        f"{'R^2':>10} ",
        f"{'||w||2':>12} ",
        f"{'Bias':>12}",
    )
    
    print("="*80)

    for alpha in alphas:
        model = RidgeRegressionScratch(alpha=alpha)
        model.fit(X_train_scaled, y_train_array)

        predictions = model.predict(X_test_scaled)

        metrics =  compute_regression_metrics(y_test_array, predictions)
        norm = weight_l2_norm(model.weights)

        print(
            f"{alpha:>10.2f} ",
            f"{metrics['RMSE']:>12.4f} ",
            f"{metrics['R^2']:>10.4f} ",
            f"{norm:>12.4f} ",
            f"{model.bias:>12.4f}"
        )


if __name__ == "__main__":
    main()