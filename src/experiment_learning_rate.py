import numpy as np

from data import load_california_housing, split_data
from preprocessing import scale_features
from linear_regression import LinearRegressionScartch
from normal_equation import LinearRegressionNormalEquation

from evaluate import compute_regression_metrics


def parameter_distance(
        gd_weights,
        gd_bias,
        ne_weights,
        ne_bias,
):
    gd_theta = np.r_[gd_bias, gd_weights]
    ne_theta = np.r_[ne_bias, ne_weights]

    return np.linalg.norm(gd_theta - ne_theta)


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


    normal_model = LinearRegressionNormalEquation()
    normal_model.fit(X_train_scaled, y_train_array)

    experiments = [
        (0.001, 1000),
        (0.005, 1000),
        (0.01, 1000),
        (0.03, 1000),
        (0.05, 1000),
        (0.01, 3000),
        (0.01, 5000),
        (0.03, 3000),
    ]

    print("\n"+"="*80)
    print("Learning rate experiment")
    print("="*80)

    print(
        f"{'LR':>8}",
        f"{'Iter':>8}",
        f"{'Final loss':>12}",
        f"{'Test RMSE':>12}",
        f"{'Test R2':>10}",
        f"{'Paraam Dist':>12}",           
    )

    print("="*80)

    for learning_rate, n_iter in experiments:
        gd_model = LinearRegressionScartch(
            learning_rate=learning_rate,
            n_iteration=n_iter,
            verbose=False)
        gd_model.fit(X_train_scaled, y_train_array)

        y_test_pred = gd_model.predict(X_test_scaled)

        metrics = compute_regression_metrics(y_test_array, y_test_pred)

        final_loss = gd_model.loss_history[-1]

        distance = parameter_distance(
            gd_model.weights,
            gd_model.bias,
            normal_model.weights,
            normal_model.bias,
        )

        print(
            f"{learning_rate:>8.3f} ",
            f"{n_iter:>8} ",
            f"{final_loss:>12.4f} ",
            f"{metrics['RMSE']:>12.4f} ",
            f"{metrics['R^2']:10.4f} ",
            f"{distance:>12.4f}",
        )

if __name__ == "__main__":
    main()

