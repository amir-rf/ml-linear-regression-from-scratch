from data import load_california_housing

def main():
    X, y = load_california_housing()

    print("="*80)
    print("Dataset Overview")
    print("="*80)

    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")

    print("\nFeature names:")
    for column in X.columns:
        print(f"- {column}")


    print("\n" + "="*80)
    print("First 5 rows of features:")
    print("="*80)
    print(X.head())
    
    print("\n" + "="*80)
    print("Missing values")
    print("="*80)
    missing_values = X.isnull().sum()
    print(missing_values)

    print("\nTotal missing values in features:", missing_values.sum())
    print("Total missing values in target:", y.isnull().sum())

    print("\n" + "="*80)
    print("Feature summary statistics")
    print("="*80)
    print(X.describe())


    print("\n" + "="*80)
    print("Target summary statistics")
    print("="*80)
    print(y.describe())


if __name__ == "__main__":
    main()



