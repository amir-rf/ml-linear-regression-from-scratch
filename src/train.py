import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

    train_mae = mean_absolute_error(y_train_array, y_train_pred)
    test_mae = mean_absolute_error(y_test_array, y_test_pred)

    train_rmse = np.sqrt(mean_squared_error(y_train_array, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test_array, y_test_pred))

    train_r2 = r2_score(y_train_array, y_train_pred)
    test_r2 = r2_score(y_test_array, y_test_pred)

    print("\n" + "="*80)
    print("Training complete")
    print("="*80)

    print("\nModel parameters:")
    print(f"Number of learned weights: {len(model.weights)}")
    print(f"Bias/intercept: {model.bias:.4f}")

    print("\nTraining performance:")
    print(f"MAE: {train_mae:.4f}")
    print(f"RMSE: {train_rmse:.4f}")
    print(f"R^2: {train_r2:.4f}")

    print("\nTesting performance:")
    print(f"MAE: {test_mae:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"R^2: {test_r2:.4f}")

    print("\nLearned feature weights:")
    for feature_name, weight in zip(X.columns, model.weights):
        print(f"{feature_name:>10}: {weight:.4f}")


if __name__ == "__main__":
    main()