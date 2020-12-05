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
from sklearn.linear_model import LogisticRegression
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

    # build preprocessor to be used by all models
    ordinal = ["age", "household_income"]
    categorical = ["us_region", "gender"]

    ordinal_pipe = make_pipeline(SimpleImputer(strategy="most_frequent"), OrdinalEncoder())
    categorical_pipe = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder())
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipe, categorical),
            ("ordinal", ordinal_pipe, ordinal),
        ],
        remainder="drop",
    )

    # List of the classifiers to be used
    classifiers = [
        DummyClassifier(strategy='stratified'),
        RandomForestClassifier(),
        LogisticRegression(max_iter = 3000),
    ]

    # Pipeline tuning settings--------------------------
    # Classifers 1 and 2 will be used within Randomized search
    param_dists = {}
    # Empty param_dist for dummy classifier
    param_dists[0] = {}
    # Param dists for classifier 1
    param_dists[1] = {
        "max_depth": scipy.stats.randint(10, 100),
        "min_samples_split": scipy.stats.randint(2, 25),
    }
    # TODO: Edit if tuning hparams for classifier 2
    # Param dists for classifier 2
    param_dists[2] = {
        "C": scipy.stats.loguniform(1e-3, 1e3)
    }

    # Settings for RandomizedSearchCV
    # TODO: Change later
    cv = 5
    scoring = "f1"
    n_iter_final = 50

    # List of classifier names for quick reference
    clf_names = []
    # Dictionary of classifiers
    model_dict = {} 
    # Build pipelines
    for i in range(len(classifiers)):
        classifier_name = str(classifiers[i]).split("(")[0]
        clf_names.append(classifier_name)
        pipe = build_pipeline(
            base_classifier=classifiers[i],
            classifier_name=clf_names[i],
            param_dists=param_dists[i],
            preprocessor=preprocessor,
            cv=cv,
            scoring=scoring,
            n_iter_final=n_iter_final
        )
        model_dict[classifier_name] = pipe
        
    # Summary Scores
    summary_score = {}
    for clf in model_dict.keys():
        model_dict[clf].fit(X_train, y_train)
        summary_score[clf] = f1_score(y_test, model_dict[clf].predict(X_test))

    # Summary table ---------------------------------------------
    # TODO: This could be done better, in a function maybe
    summary_df = pd.DataFrame(
        data=[np.round(summary_score["LogisticRegression"], 3), 
        np.round(summary_score["RandomForestClassifier"], 3), 
        np.round(summary_score["DummyClassifier"], 3)],
        index=["LogisticRegression", "RandomForest", "DummyClassifier"],
        columns=["F1 Score"],
    )
    base_classifier = classifiers[1]

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

    #  ROC Plots and Confusion matrixfor all models
    for classifier_name in clf_names:
        build_roc_plot(
            model_dict[classifier_name],
            classifier_name=classifier_name,
            X=X_test,
            y=y_test,
            visuals_path=visuals_path,
        )
        build_confusion_matrix_plot(
            model_dict[classifier_name],
            classifier_name=classifier_name,
            X=X_test,
            y=y_test,
            visuals_path=visuals_path,
        )
        if classifier_name == "DummyClassifier":
            continue
        build_shap_plot(
            model_dict[classifier_name],
            classifier_name=classifier_name,
            X=X_train,
            y=y_train,
            visuals_path=visuals_path,
        )

    return 




def build_pipeline(
    base_classifier: ClassifierMixin,
    classifier_name : str,
    param_dists: dict,
    preprocessor, #sklearn.compose._column_transformer.ColumnTransformer,
    cv: int = 5,
    scoring: str = "f1",
    n_iter_final: int = 10,
) -> Pipeline:
    """
    Builds a pipeline with tuned hyperparameters for a given classifier and a ColumnTransformer
    using RandomizedSearchCV over the param dists passed in for the classifier.

    Parameters
    ----------
    base_classifier: sklearn.base.Classifier
        a sklearn classifier object
    classifier_name : str
        name of classifier
    param_dists: dict
        param distributions compatible with the classifier passed in to be used in RandomizedSearchCV
    preprocessor: sklearn.compose._column_transformer.ColumnTransformer
        an sklearn columntransformer object to be include in the pipeline
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
    
    # This pipeline will run the preprocessing and
    # uses RandomizedSearchCV search of hyperparameters
    # on the specified classifier
        
    # If DummyClassifier, don't do any tuning
    if classifier_name == "DummyClassifier":
        main_pipe = make_pipeline(preprocessor, base_classifier)
    else:
        main_pipe = make_pipeline(
            preprocessor,
            RandomizedSearchCV(
                estimator=base_classifier,
                param_distributions=param_dists,
                cv=cv,
                scoring=scoring,
                refit=scoring,
                n_iter=n_iter_final,
                verbose=1,
                n_jobs=-2,
            )
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
    y : pd.DataFrame
        target column for Classifier
    visuals_path: str
        path to save plot to.
    """

    #TODO: add more descriptive comments and abstract away some objects into arguments or global constants
    
    preprocessor = classifier.named_steps['columntransformer']

    preprocessor.fit(X, y)

    feature_names = get_column_names_from_ColumnTransformer(preprocessor)

    X_enc = pd.DataFrame(
        data=preprocessor.transform(X),
        columns=feature_names,
        index=X.index,
    )
    if classifier_name == "LogisticRegression":
        # TODO: fix this
        X_enc = shap.sample(X_enc, 20)
        explainer = shap.KernelExplainer(classifier.named_steps['randomizedsearchcv'].best_estimator_.predict_proba, data=X_enc)
    else:
        explainer = shap.TreeExplainer(classifier.named_steps['randomizedsearchcv'].best_estimator_)

    fig, ax = plt.subplots()

    shap_values = np.array(explainer.shap_values(X_enc))

    shap.summary_plot(
        shap_values[-1], X_enc, show=False
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
