"""Compatibility class required to unpickle the dissertation StackedEnsemble model."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict


class WeightedOOFStackingClassifier(ClassifierMixin, BaseEstimator):
    """
    Leakage-controlled two-level stacked classifier used by the final pipeline.

    Base signals:
      - SVM: decision_function
      - Random Forest: predict_proba
    Meta learner:
      - XGBoost (or compatible final estimator)
    """

    def __init__(
        self,
        svm_estimator,
        rf_estimator,
        final_estimator,
        high_risk_class_encoded,
        high_risk_weight=1.0,
        cv_splits=5,
        random_state=42,
        n_jobs=-1,
    ):
        self.svm_estimator = svm_estimator
        self.rf_estimator = rf_estimator
        self.final_estimator = final_estimator
        self.high_risk_class_encoded = high_risk_class_encoded
        self.high_risk_weight = high_risk_weight
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _make_cv(self):
        return StratifiedKFold(
            n_splits=self.cv_splits,
            shuffle=True,
            random_state=self.random_state,
        )

    @staticmethod
    def _as_2d(arr):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def fit(self, X, y):
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)

        svm_oof = cross_val_predict(
            clone(self.svm_estimator),
            X,
            y_arr,
            cv=self._make_cv(),
            method="decision_function",
            n_jobs=self.n_jobs,
        )
        rf_oof = cross_val_predict(
            clone(self.rf_estimator),
            X,
            y_arr,
            cv=self._make_cv(),
            method="predict_proba",
            n_jobs=self.n_jobs,
        )

        meta_X = np.hstack([self._as_2d(svm_oof), self._as_2d(rf_oof)])
        sample_weight = np.where(
            y_arr == self.high_risk_class_encoded,
            float(self.high_risk_weight),
            1.0,
        )

        self.final_estimator_ = clone(self.final_estimator)
        try:
            self.final_estimator_.fit(meta_X, y_arr, sample_weight=sample_weight)
        except TypeError:
            self.final_estimator_.fit(meta_X, y_arr)

        self.svm_estimator_ = clone(self.svm_estimator)
        self.rf_estimator_ = clone(self.rf_estimator)
        self.svm_estimator_.fit(X, y_arr)
        self.rf_estimator_.fit(X, y_arr)
        return self

    def _meta_features(self, X):
        svm_signal = self._as_2d(self.svm_estimator_.decision_function(X))
        rf_signal = self._as_2d(self.rf_estimator_.predict_proba(X))
        return np.hstack([svm_signal, rf_signal])

    def predict(self, X):
        return self.final_estimator_.predict(self._meta_features(X))

    def predict_proba(self, X):
        meta_X = self._meta_features(X)
        if hasattr(self.final_estimator_, "predict_proba"):
            return self.final_estimator_.predict_proba(meta_X)

        decision = np.asarray(self.final_estimator_.decision_function(meta_X))
        if decision.ndim == 1:
            decision = np.column_stack([-decision, decision])
        decision = decision - np.max(decision, axis=1, keepdims=True)
        exp_decision = np.exp(decision)
        return exp_decision / exp_decision.sum(axis=1, keepdims=True)
