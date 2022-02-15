from . import calibration_curves as cc
import pandas as pd


class ConcentrationEstimator:
    def __init__(self):
        """
        Please add docstrings...
        """
        self.params_ = pd.DataFrame()
        self.interval = [0.5, 2]

    def set_interval(self, interval):
        self.interval = interval

    def fit(self, X, y, v_slope="fixed"):
        if v_slope == "interval":
            self.params_ = cc.calibration_curves_variable_slope_interval(
                X, y, self.interval
            )
        if v_slope == "wide":
            self.params_ = cc.calibration_curves_variable_slope(X, y)
        if v_slope == "fixed":
            self.params_ = cc.calibration_curves(X, y)

    def predict(self, X):
        return cc.transform(X, self.params_)

    def score(self, y_pred, y_true):
        # X is the matrix of standard curves.....
        return cc.fitting_score(y_pred, y_true)
