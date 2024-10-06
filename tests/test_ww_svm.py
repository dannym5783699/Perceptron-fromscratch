from pathlib import Path

import pandas as pd
from assignment2_neuralnetworks.classifier_wrapper import ClassifierWrapper
from assignment2_neuralnetworks.models import WestonWatkinsSVM
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    cv_results_directory = Path(__file__).parent.parent / "cv_results"

    x_matrix, y = load_iris(return_X_y=True)
    n_classes = len(set(y))

    param_grid = {
        'classifier__model_class': [WestonWatkinsSVM],
        'classifier__n_features': [x_matrix.shape[1]],
        'classifier__n_classes': [n_classes],
        'classifier__lr': [0.1, 1e-2, 1e-4, 1e-8],
        'classifier__max_epochs': [5, 10, 50, 100, 200, 500, 1000],
        'classifier__binary': [False]
    }

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', ClassifierWrapper(model_class=None))
    ])

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, verbose=1, n_jobs=-1)

    print(f"Running CV for Weston Watkins SVM")
    grid_search.fit(x_matrix, y)
    print("CV complete")

    save_results = pd.DataFrame(grid_search.cv_results_)
    results_fpath = cv_results_directory / f"cv_results_ww_svm.csv"

    print("Saving results to", results_fpath)

    save_results.to_csv(results_fpath, index=False)


if __name__ == "__main__":
    main()
