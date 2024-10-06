from pathlib import Path

import pandas as pd
from classifier_wrapper import ClassifierWrapper
from models import WestonWatkinsSVM
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":

    cv_results_directory = Path(__file__).parent.parent / "cv_results"

    X, y = load_iris(return_X_y=True)
    n_classes = len(set(y))


    param_grid = {
        'classifier__model_class': [WestonWatkinsSVM],
        'classifier__n_features': [X.shape[1]],
        'classifier__n_classes': [n_classes],
        'classifier__lr': [0.1, 1e-2, 1e-4, 1e-8],
        'classifier__max_epochs': [5, 10, 50, 100, 200, 500, 1000],
        'classifier__binary': [False]
    }

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', ClassifierWrapper(model_class=None))
    ])

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, verbose=1, n_jobs=-1)


    print(f"Running CV for Weston Watkins SVM")
    grid_search.fit(X, y)
    print("CV complete")

    save_results = pd.DataFrame(grid_search.cv_results_)
    results_fpath = cv_results_directory / f"cv_results_ww_svm.csv"

    print("Saving results to", results_fpath)

    save_results.to_csv(results_fpath, index=False)


