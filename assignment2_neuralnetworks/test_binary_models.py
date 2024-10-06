from pathlib import Path

import pandas as pd
from classifier_wrapper import ClassifierWrapper
from models import LinearSVM, LogisticRegression, WidrowHoff
from sklearn.datasets import fetch_openml, load_breast_cancer, load_iris
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_iris_binary():
    X_iris, y_iris = load_iris(return_X_y=True)
    # Make the iris dataset binary by removing the class with label 2
    X_iris = X_iris[y_iris != 2]
    y_iris = y_iris[y_iris != 2]
    # Convert the labels to binary
    y_iris = (y_iris == 1).astype(int)
    return X_iris, y_iris

def get_titanic():
    """https://openml.org/search?type=data&status=active&sort=nr_of_downloads&id=42438"""
    X, y = fetch_openml(data_id=42438, return_X_y=True)
    return X, y

if __name__ == "__main__":

    cv_results_directory = Path(__file__).parent.parent / "cv_results"

    datasets = {
        "breast_cancer": load_breast_cancer(return_X_y=True),
        "iris": get_iris_binary(),
        "titanic" : get_titanic()
    }
    
    LEARNING_RATES = [1e-2, 1e-4, 1e-8]
    MAX_EPOCHS = [10, 50, 100, 500]

    param_grid = [
        # Parameter grid for WidrowHoff (which needs n_features)
        {
            'classifier__model_class': [WidrowHoff],
            'classifier__n_features': None,  # Only for WidrowHoff
            'classifier__lr': LEARNING_RATES,
            'classifier__max_epochs': MAX_EPOCHS
        },
        # Parameter grid for models that do not need n_features
        {
            'classifier__model_class': [LinearSVM, LogisticRegression],
            'classifier__lr': LEARNING_RATES,
            'classifier__max_epochs': MAX_EPOCHS
        }
    ]

    pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('classifier', ClassifierWrapper(model_class=None))
    ])

    # Initialize GridSearchCV with the ClassifierWrapper and the parameter grid
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, verbose=1, n_jobs=-1)

    for dataset_name, (X, y) in datasets.items():
        # Set n_features for WidrowHoff
        grid_search.param_grid[0]['classifier__n_features'] = [X.shape[1]]

        print(f"Running CV on {dataset_name}")
        grid_search.fit(X, y)
        print("CV complete")

        save_results = pd.DataFrame(grid_search.cv_results_)
        results_fpath = cv_results_directory / f"cv_results_{dataset_name}.csv"

        print("Saving results to", results_fpath)

        save_results.to_csv(results_fpath, index=False)


