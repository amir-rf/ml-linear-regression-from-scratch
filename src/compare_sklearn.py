import numpy as np

from sklearn.linear_model import LinearRegression, Ridge

from data import load_california_housing, split_data
from preprocessing import scale_features
from evaluate import compute_regression_metrics
from normal_equation import LinearRegressionNormalEquation
from ridge_regression import RidgeRegressionScratch



def print_metrics(model_name, metrics):
    print("\n"+"="*80)
    print(model_name)
    print("="*80)

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")


def main():
    X, y = load_california_housing()
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    y_train_array = y_train.to_numpy()
    y_test_array = y_test.to_numpy()

    scratch_linear = LinearRegressionNormalEquation()
    scratch_linear.fit(X_train_scaled, y_train_array)
    scratch_linear_pred = scratch_linear.predict(X_test_scaled)


    sklearn_linear = LinearRegression()
    sklearn_linear.fit(X_train_scaled, y_train_array)
    sklearn_linear_pred = sklearn_linear.predict(X_test_scaled)


    scratch_ridge = RidgeRegressionScratch(alpha=100)
    scratch_ridge.fit(X_train_scaled, y_train_array)
    scratch_ridge_pred = scratch_ridge.predict(X_test_scaled)

    sklearn_ridge = Ridge(alpha=100)
    sklearn_ridge.fit(X_train_scaled, y_train_array)
    sklearn_ridge_pred = sklearn_ridge.predict(X_test_scaled)


    print_metrics(
        "Scratch linear regression",
        compute_regression_metrics(y_test_array, scratch_linear_pred)
    )
    print_metrics(
        "sklearn Linear Regression",
        compute_regression_metrics(y_test_array, sklearn_linear_pred)
    )

    print_metrics(
        "Scratch ridge regression alpha=100",
        compute_regression_metrics(y_test_array, scratch_ridge_pred)
    )

    print_metrics(
        "sklearn ridge regression alpha=100",
        compute_regression_metrics(y_test_array, sklearn_ridge_pred)
    )


    print("\n"+"="*80)
    print("Coefficient check")
    print("="*80)

    print("\nLinear regression coefficent differences:")

    for feature_name, scratch_w, sklearn_w in zip(
        X.columns,
        scratch_linear.weights,
        sklearn_linear.coef_,
    ):
        diff = scratch_w - sklearn_w
        print(f"{feature_name:>10}: diff={diff: .8f}")


    print("\nRidge regression coefficient differences:")

    for feature_name, scratch_w, sklearn_w in zip(
        X.columns,
        scratch_ridge.weights,
        sklearn_ridge.coef_,
    ):
        diff = scratch_w - sklearn_w
        print(
            f"{feature_name:>10}: diff={diff: .8f}"
        )


if __name__ == "__main__":
    main()
