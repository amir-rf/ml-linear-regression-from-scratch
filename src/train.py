
from evaluate import (
    compute_regression_metrics,
    plot_loss_curve,
    plot_prediction_vs_actual,
    plot_residual_histogram,
)
from data import load_california_housing, split_data
from linear_regression import LinearRegressionScartch
from preprocessing import scale_features

def main():

    X, y = load_california_housing()

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42,)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    y_train_array = y_train.to_numpy()
    y_test_array = y_test.to_numpy()

    model = LinearRegressionScartch(
        learning_rate=0.01, n_iteration=1000,
    )

    model.fit(X_train_scaled, y_train_array)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)


    train_metrics = compute_regression_metrics(y_train_array, y_train_pred)
    test_metrics = compute_regression_metrics(y_test_array, y_test_pred)

    print("\n" + "="*80)
    print("Training complete")
    print("="*80)

    print("\nModel parameters:")
    print(f"Number of learned weights: {len(model.weights)}")
    print(f"Bias/intercept: {model.bias:.4f}")

    print("\nTraining performance:")
    for metric_name, metric_value in train_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    for metric_name, metric_value in test_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
        
    print("\nLearned feature weights:")
    for feature_name, weight in zip(X.columns, model.weights):
        print(f"{feature_name:>10}: {weight:.4f}")

    
    plot_loss_curve(model.loss_history)
    plot_prediction_vs_actual(y_test_array, y_test_pred)
    plot_residual_histogram(y_test_array, y_test_pred)

    print("\nSaved diagnostic plots to the figures/ folder.")


if __name__ == "__main__":
    main()