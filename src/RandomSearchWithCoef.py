from sklearn.model_selection import RandomizedSearchCV

# -------------------------------------------------
class RandomizedSearchWithCoef(RandomizedSearchCV):
    @property
    def coef_(self):
        return self.best_estimator_.coef_

    @property
    def feature_importances_(self):
        return self.best_estimator_.feature_importances_