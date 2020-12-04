# author: Dustin Andrews - Group 11
# date: 2020-11-26
"""
This script will read in a preprocessed data file from the fivethirtyeight Earthquake dataset, do some minor preprocessing, then run a modelling process to 
build a classifier of whether a person is afraid of earthquakes or not.

Usage: docopt.py --input_train_file_path=<input_file_path> --input_test_file_path=<input__test_file_path>  --output_visuals_path=<output_visuals_path>

Options:
--input_train_file_path=<input_train_file_path>     String: path to input training data file
--input_test_file_path=<input__test_file_path>     String: path to input test data file
--output_visuals_path=<output_visuals_path>      String: path to folder where script will write model diagnostic plots and present results. 
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.numeric import cross
import pandas as pd
import random


from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import ClassifierMixin


# other
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    plot_roc_curve,
    plot_confusion_matrix,
    classification_report,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
)
from sklearn.impute import SimpleImputer

# For sampling distributions
import scipy.stats

# For Shapley Addtive Explanations (SHAP)
import shap

# Documentation
from docopt import docopt
import typing

from utils import get_column_names_from_ColumnTransformer

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
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        raise Exception("Read file unsuccessful, check the file_path")
    except ValueError:
        raise Exception("Ensure file path is a string")

    return data


def run_modelling(
    training_set: pd.DataFrame, test_set: pd.DataFrame, visuals_path: str
) -> None:
    """Runs classifier 1 (RandomForestClassifier) and classifier 2 (LogisticRegression) and compares against a baseline of DummyClassifier on the
    Earthquake training data set.

    Creates ROC curve visual for all classifiers, Confusion Matrix for all classifiers and SHAP summary plots for both classifiers 1 & 2.

    Parameters
    ----------
    training_set: (pd.DataFrame):
        training set split of Earthquake data, used with cross validation to determine best model
    test_set : (pd.DataFrame):
        test set split of Earthquake data, used to evaluate the final model
    visuals_path (str):
        file path to write visuals out to

    # TODO: Specify return type (Pipeline for best performing model)
    """
    # For reproducibility set seed
    random.seed(42)

    # Build dataframes for training
    X_train, y_train = (
        training_set.drop(columns=["target"]),
        training_set.loc[:, "target"],
    )

    X_test, y_test = (
        test_set.drop(columns=["target"]),
        test_set.loc[:, "target"],
    )

    # List of classifiers and model to iterate over
    clf_names = ["DummyClassifier", "RandomForestClassifier"]
    clf_list = [DummyClassifier(strategy = 'stratified'), RandomForestClassifier()]

    # Pipeline tuning settings--------------------------
    # Classifers 1 and 2 will be used within Randomized search
    param_dists = {}
    # Empty param_dist for dummy classifier
    param_dists[0] = {}
    # Settings specific to classifier 1
    param_dists[1] = {
        "max_depth": scipy.stats.randint(10, 100),
        "min_samples_split": scipy.stats.randint(2, 25),
    }

    # Settings for RandomizedSearchCV
    cv = 5
    scoring = "f1"
    n_iter_final = 50

    # List of pipelines to iterate over
    pipe_list = []
    # Build pipelines
    for i in range(len(clf_names)):
        pipe = build_pipeline(
            base_classifier=clf_list[i],
            classifier_name=clf_names[i],
            param_dists=param_dists[i],
            cv=cv,
            scoring=scoring,
            n_iter_final=n_iter_final
        )
        pipe.fit(X_train, y_train)
        pipe_list.append(pipe)

    # Dictionary to match different models and pipes
    model_dict = {clf_names[i] : pipe_list[i] for i in range(len(clf_names))}

    # Summary Scores
    summary_score = {}
    for clf in model_dict.keys():
        summary_score[clf] = f1_score(y_test, model_dict[clf].predict(X_test))

    # Summary table ---------------------------------------------
    # TODO: This could be done better, in a function maybe
    summary_df = pd.DataFrame(
        data=[np.round(summary_score[clf_names[1]], 3), np.round(summary_score[clf_names[0]], 3)],
        index=[clf_names[1], clf_names[0]],
        columns=["F1 Score"],
    )
    base_classifier = clf_list[1]

    # Write out matplotlib table
    fig, ax = plt.subplots(figsize=(3, 2))
    fig.patch.set_visible(False)
    ax.axis("off")
    ax.axis("tight")

    ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        rowLabels=summary_df.index,
        colWidths=[0.25],
        loc="center",
    )
    fig.savefig(
        os.path.join(visuals_path, "classifier_results_table.png"),
        bbox_inches="tight",
    )

    # Build out plots to save----------------------------------------------------------
    classifier_type = str(clf_names[1])

    #  ROC Plots for real classifier, and benchmark Dummy
    for i in range(len(model_dict.keys())):
        build_roc_plot(
            pipe_list[i],
            classifier_name=clf_names[i],
            X=X_test,
            y=y_test,
            visuals_path=visuals_path,
        )

    # Feature Importance----------------------------------------------------------------
    # Feature Importance is only possible for Random Forest
    rf_pipe = pipe_list[1]

    feat_list = get_column_names_from_ColumnTransformer(
        rf_pipe.named_steps["preprocess"]
    )
    feat_imps = rf_pipe.named_steps["clf"].best_estimator_.feature_importances_

    feat_imp_df = pd.DataFrame(
        index=feat_list, data=feat_imps, columns=["Feature Importance %"]
    ).sort_values(by="Feature Importance %")

    fig, ax = plt.subplots()
    ax.barh(
        feat_imp_df.index,
        feat_imp_df["Feature Importance %"],
    )
    ax.set_title(f"Feature Importance for {classifier_type}")
    ax.set_xlabel("% Importance")
    fig.savefig(
        os.path.join(visuals_path, "feature_importance.png"),
        bbox_inches="tight",
    )

    for i in range(len(clf_names)):
        # Confusion Matrix for real classifier, and benchmark Dummy--------------
        build_confusion_matrix_plot(
            pipe_list[i],
            classifier_name=clf_names[i],
            X=X_test,
            y=y_test,
            visuals_path=visuals_path,
        )
        if clf_names[i] == "DummyClassifier":
            continue
        build_shap_plot(
            pipe_list[i],
            classifier_name=clf_names[i],
            X=X_train,
            y=y_train,
            visuals_path=visuals_path,
        )

    # TODO: decide whether to return the best performing model. Right now, returns the Random Forest pipeline 
    return clf[1]


def build_pipeline(
    base_classifier: ClassifierMixin,
    classifier_name : str,
    param_dists: dict,
    cv: int = 5,
    scoring: str = "f1",
    n_iter_final: int = 10,
) -> Pipeline:
    """
    Build a pipeline for the Earthquake dataset with four demographic features.
    Creates preprocessing, and then RandomizedSearchCV over the param dists passed in for the classifier.

    Parameters
    ----------
    base_classifier: sklearn.base.Classifier
        a sklearn classifier object
    classifier_name : str
        name of classifier
    param_dists: dict
        param distributions compatible with the classifier passed in to be used in RandomizedSearchCV
    cv: int
        number of cv folds to be used in RandomizedSearchCV step
    scoring: string
        scoring to use in RandomizedSearchCV
    n_iter_final: int
        number of iterations to use in RandomizedSearchCV


    Returns
    -------
    Pipeline:
        a sklearn pipeline ready to be fit
    """
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

    # This pipeline will run the preprocessing and
    # uses RandomizedSearchCV search of hyperparameters
    # on the specified classifier
    #
    # If DummyClassifier passed in, don't do any tuning
    if classifier_name == "DummyClassifier":
        main_pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "clf",
                    base_classifier,
                ),
            ]
        )
    else:
        main_pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "clf",
                    RandomizedSearchCV(
                        estimator=base_classifier,
                        param_distributions=param_dists,
                        cv=cv,
                        scoring=scoring,
                        refit=scoring,
                        n_iter=n_iter_final,
                        verbose=1,
                        n_jobs=-2,
                    ),
                ),
            ]
        )

    return main_pipe


def build_roc_plot(
    classifier: ClassifierMixin,
    classifier_name: str,
    X: pd.DataFrame,
    y: pd.DataFrame,
    visuals_path: str,
) -> None:
    """Build a Receiver Operating Characteristic Plot for classifier and dataset.
    Save plot out to specified folder

    Parameters
    ----------
    classifier : ClassifierMixin
        A fitted sklearn Classifier
    classifier_name : str
        name to use on plot for Classifier
    X : pd.DataFrame
        compatible input dataset with Classifier
    y : pd.DataFrame
        target column for Classifier
    visuals_path: str
        path to save plot to.
    """
    fig, ax = plt.subplots()
    plot_roc_curve(classifier, X, y, ax=ax)
    ax.set_title(f"Receiver Operating Characteristic Curve on {classifier_name}")
    fig.savefig(
        os.path.join(visuals_path, f"roc_auc_curve_{classifier_name}.png"),
        bbox_inches="tight",
    )


def build_confusion_matrix_plot(
    classifier: ClassifierMixin,
    classifier_name: str,
    X: pd.DataFrame,
    y: pd.DataFrame,
    visuals_path: str,
) -> None:
    """Build a Confusion Matrix for classifier and dataset.
    Save plot out to specified folder

    Parameters
    ----------
    classifier : ClassifierMixin
        A fitted sklearn Classifier
    classifier_name : str
        name to use on plot for Classifier
    X : pd.DataFrame
        compatible input dataset with Classifier
    y : pd.DataFrame
        target column for Classifier
    visuals_path: str
        path to save plot to.
    """
    fig, ax = plt.subplots()
    ax.grid(False)
    plot_confusion_matrix(classifier, X, y, ax=ax)
    ax.set_title(f"Confusion Matrix on {classifier_name}")
    fig.savefig(
        os.path.join(visuals_path, f"confusion_matrix_{classifier_name}.png"),
        bbox_inches="tight",
    )


def build_shap_plot(
    classifier: ClassifierMixin,
    classifier_name: str,
    X: pd.DataFrame,
    y: pd.DataFrame,
    visuals_path: str,
) -> None:
    """Build a Shapley Additive Explanations (SHAP) Summary Plot for classifier and dataset.
    Save plot out to specified folder

    Parameters
    ----------
    classifier : ClassifierMixin
        A fitted sklearn Classifier
    classifier_name : str
        name to use on plot for Classifier
    X : pd.DataFrame
        compatible input dataset with Classifier
    visuals_path: str
        path to save plot to.
    """

    #TODO: add more descriptive comments and abstract away some objects into arguments or global constants

    explainer = shap.TreeExplainer(classifier.named_steps['clf'].best_estimator_)

    preprocessor = classifier.named_steps['preprocess']

    ohe_feature_names = (
        preprocessor.named_transformers_["categorical"]
        .named_steps["one_hot"]
        .get_feature_names()
        .tolist()
    )

    ordinal = ["age", "household_income"]

    feature_names = (
        ohe_feature_names + ordinal
    )

    X_enc = pd.DataFrame(
        data=preprocessor.transform(X),
        columns=feature_names,
        index=X.index,
    )

    fig, ax = plt.subplots()

    shap_values = np.array(explainer.shap_values(X_enc))

    shap.summary_plot(
        shap_values[-1], X_enc
    )
    ax.set_title(f"SHAP Summary Plot on {classifier_name}")
    fig.savefig(
        os.path.join(visuals_path, f"shap_summary_plot_{classifier_name}.png"),
        bbox_inches="tight",
    )

if __name__ == "__main__":
    opt = docopt(__doc__)

    training_set = get_data(opt["--input_train_file_path"])
    test_set = get_data(opt["--input_test_file_path"])

    best_model = run_modelling(
        training_set=training_set,
        test_set=test_set,
        visuals_path=opt["--output_visuals_path"],
    )
