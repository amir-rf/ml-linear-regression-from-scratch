from data import load_california_housing, split_data
from evaluate import compute_regression_metrics
from feature_engineering import add_engineered_features
from normal_equation import LinearRegressionNormalEquation
from preprocessing import scale_features



def train_and_evaluate(X, y, label):
    X_train,  X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    y_train_array = y_train.to_numpy()
    y_test_array = y_test.to_numpy()

    model = LinearRegressionNormalEquation()
    model.fit(X_train_scaled, y_train_array)

    predictions = model.predict(X_test_scaled)

    metrics = compute_regression_metrics(y_test_array, predictions)

    print("\n"+"="*80)
    print(label)
    print("="*80)

    print(f"Number of features: {X.shape[1]}")

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:>.4f}")

    return metrics


def main():
    X, y = load_california_housing()
    train_and_evaluate(X, y, label="Baseline features")

    X_engineered = add_engineered_features(X)
    train_and_evaluate(X_engineered, y, label="Baseline + engineered features")


if __name__ == "__main__":
    main()