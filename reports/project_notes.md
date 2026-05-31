## Model Experiments and Lessons

### 1. Gradient Descent Linear Regression

We implemented linear regression from scratch using NumPy and gradient descent.

The model minimizes mean squared error by repeatedly updating the weights in the opposite direction of the gradient.

Key observation:

- Loss decreased smoothly.
- Gradient descent converged, but 1000 iterations with learning rate 0.01 did not fully reach the optimal solution.
- Increasing the learning rate improved convergence speed.

Best tested gradient descent setting:

- learning rate: 0.03
- iterations: 3000
- parameter distance from Normal Equation: approximately 0.0074

### 2. Normal Equation

We implemented the analytical solution for ordinary least squares.

The Normal Equation directly solves:

\[
w = (X^T X)^{-1}X^T y
\]

In practice, we used the pseudo-inverse for numerical stability.

Key observation:

- Normal Equation gave slightly better performance than early gradient descent.
- It served as a reference solution for checking whether gradient descent had converged.

### 3. Ridge Regression

We implemented Ridge Regression from scratch using the closed-form solution:

\[
w = (X^T X + \lambda I)^{-1}X^T y
\]

Ridge adds an L2 penalty to discourage large coefficients.

Key observation:

- Moderate regularization improved test performance slightly.
- Very large regularization caused underfitting.
- The weight norm decreased as alpha increased.

Best tested Ridge setting:

- alpha: 100
- RMSE: approximately 0.7438
- R²: approximately 0.5778

### 4. scikit-learn Verification

We compared our scratch implementations with scikit-learn.

Results:

- Scratch Linear Regression matched sklearn LinearRegression.
- Scratch Ridge Regression matched sklearn Ridge.
- Coefficient differences were effectively zero.

This confirms that our implementations are mathematically and programmatically correct.

### 5. Model Limitations

Linear models are limited because they assume an additive linear relationship between features and target.

Observed limitations:

- Expensive houses are often underpredicted.
- The target appears capped at the upper end.
- Linear regression cannot automatically model nonlinear relationships or feature interactions.

Examples of possible feature interactions:

- income × location
- rooms × location
- house age × income

