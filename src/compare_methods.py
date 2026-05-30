from data import load_california_housing, split_data
from evaluate import compute_regression_metrics
from linear_regression import LinearRegressionScartch
from normal_equation import LinearRegressionNormalEquation
from preprocessing import scale_features


def print_metrics(model_name, metrics):

    print("\n"+"="*80)
    print(model_name)
    print("\n"+"="*80)

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

    gd_model = LinearRegressionScartch(
        learning_rate=0.01,
        n_iteration=1000,
            )
    
    gd_model.fit(X_train_scaled, y_train_array)

    ne_model = LinearRegressionNormalEquation()

    ne_model.fit(X_train_scaled, y_train_array)

    gd_predictions = gd_model.predict(X_test_scaled)
    ne_predictions = ne_model.predict(X_test_scaled)

    gd_metrics = compute_regression_metrics(y_test_array, gd_predictions)
    ne_metrics = compute_regression_metrics(y_test_array, ne_predictions)

    print_metrics("Gradient descent linear regression", gd_metrics)
    print_metrics("Normal equiation linear regression", ne_metrics)

    print("\n"+"="*80)
    print("Parameter comparison")
    print("\n"+"="*80)

    print(f"Gradient descent bias: {gd_model.bias:.4f}")
    print(f"Normal equiation bias: {ne_model.bias:.4f}")

    print("\nFeature weights:")
    for feature_name, gd_weight, ne_weight in zip(
        X.columns,
        gd_model.weights,
        ne_model.weights,
    ):
        difference = gd_weight - ne_weight
        print(
            f"{feature_name:>10}: "
            f"GD={gd_weight:.4f}, "
            f"NE={ne_weight:.4f}, "
            f"Diff={difference:.4f}"
        )
        

if __name__ == "__main__":
    main()
