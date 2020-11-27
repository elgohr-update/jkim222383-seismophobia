# author: Dustin Andrews - Group 11
# date: 2020-11-26
"""
This script will read in a preprocessed data file from the fivethirtyeight Earthquake dataset, do some minor preprocessing, then run a modelling process to 
build a classifier of whether a person is afraid of earthquakes or not.

Usage: docopt.py --input_file_path=<input_file_path> --output_visuals_path=<output_visuals_path>

Options:
--input_file_path=<input_file_path>     String: path to input data file
--output_visual_path=<output_visuals_path>      String: path to write model diagnostic plots out to and present results. 
"""

from math import remainder
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

# data
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# Feature selection
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer

# other
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    plot_roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
)
from sklearn.svm import SVC, SVR

# Custom Random Search Class
from RandomSearchWithCoef import RandomizedSearchWithCoef

# For sampling distributions
import scipy.stats

# Documentation
from docopt import docopt
import typing

# Graph styling setup
plt.style.use("seaborn")


def get_data(file_path: str) -> pd.DataFrame:
    """Gets the Earthquake data set, with renamed columns

    Parameters
    ----------
    file_path: str
        Path to the earthquake data set

    Returns
    -------
    pandas.DataFrame:
        A pandas dataframe of the data
    """
    if isinstance(file_path, str) != True:
        print("Ensure the file path is numeric")
        return

    try:
        data = pd.read_csv(file_path)
    except:
        raise Exception("Read file unsuccessful, check the file_path")

    return data


def run_modelling(training_set: pd.DataFrame, visuals_path: str) -> None:
    """Runs random forest classifier, support vector classifier and logistic regression classifier over the
    Earthquake training data set

    Parameters
    ----------
    training_set: (pd.DataFrame):
        training set split of Earthquake data, used with cross validation to determine best model
    visuals_path (str):
        file path to write visuals out to
    """
    # For reproducibility set seed
    random.seed(42)

    # Build dataframes for training
    X_train, y_train = (
        training_set.drop(columns=["target"]),
        training_set.loc[:, "target"],
    )

    # For our dataset, just the two types of variables are used.
    ordinal = ["age", "household_income"]
    categorical = ["us_region", "gender"]

    ordinal_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal_encode", OrdinalEncoder()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipe, categorical),
            ("ordinal", ordinal_pipe, ordinal),
        ],
        remainder="drop",
    )

    # TODO: MOVE THIS TO A FUNCTION! Take in classifier, param_grid
    # Pipeline tuning settings--------------------------
    # Classifer will be used within RFECV, and Randomized search
    base_classifier = RandomForestClassifier()

    # Settings specific to classifier chosen.
    param_dists = {
        "max_depth": scipy.stats.randint(10, 100),
        "min_samples_split": scipy.stats.randint(2, 25),
    }

    cv = 2
    scoring = "f1"

    # Number of iterations when doing feature search
    n_feature_search = 2
    # Number of iterations when doing final model tuning on feature set
    n_iter_final = 10

    # This pipeline will run the preprocessing, then do feature selection with RandomizedSearchCV
    # within each set of features. Finally it uses that feature set in a longer search of hyperparameters
    # on the specified classifier
    main_pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "feature_select",
                RFECV(
                    RandomizedSearchWithCoef(
                        estimator=base_classifier,
                        param_distributions=param_dists,
                        cv=cv,
                        scoring=scoring,
                        refit=scoring,
                        n_iter=n_feature_search,
                        n_jobs=1,
                    ),
                    cv=cv,
                    scoring=scoring,
                    verbose=1,
                    n_jobs=-2,
                ),
            ),
            (
                "clf",
                RandomizedSearchCV(
                    estimator=base_classifier,
                    param_distributions=param_dists,
                    cv=cv,
                    scoring=scoring,
                    refit=scoring,
                    n_iter=n_iter_final,
                ),
            ),
        ]
    )

    main_pipe.fit(X_train, y_train)

    # Summary Scores
    cv_f1_score = cross_val_score(main_pipe, X_train, y_train, scoring=scoring, cv=cv)
    print(f"Final F1 CV Score: {np.mean(cv_f1_score):.3f}")

    # Build out plots to save
    auc_roc_plot = plot_roc_curve(main_pipe, X_train, y_train)
    auc_roc_plot.figure_.savefig(
        f"{visuals_path}/roc_auc_curve.png", bbox_inches="tight"
    )

    return main_pipe


if __name__ == "__main__":
    # opt = docopt(__doc__)
    # training_data = get_data(opt["--input_file_path"])
    # os.chdir("..")
    training_set = get_data("data/processed/earthquake_data_train.csv")
    best_model = run_modelling(training_set=training_set, visuals_path="visuals")