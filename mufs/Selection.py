from math import sqrt
from sys import float_info
from itertools import combinations
import numpy as np
from .Metrics import Metrics
from ._version import __version__


class MUFS:
    """Compute Fast Fast Correlation Based Filter
    Yu, L. and Liu, H.; Feature Selection for High-Dimensional Data: A Fast
    Correlation Based Filter Solution,Proc. 20th Intl. Conf. Mach. Learn.
    (ICML-2003)

    and

    Correlated Feature Selection as in "Correlation-based Feature Selection for
    Machine Learning" by Mark A. Hall

    Parameters
    ----------
    max_features: int
        The maximum number of features to return
    discrete: boolean
        If the features are continuous or discrete. It always supose discrete
        labels.
    """

    def __init__(self, max_features=None, discrete=True):
        self.max_features = max_features
        self._discrete = discrete
        self.symmetrical_uncertainty = (
            Metrics.symmetrical_uncertainty
            if discrete
            else Metrics.symmetrical_unc_continuous
        )
        self.symmetrical_uncertainty_features = (
            Metrics.symmetrical_uncertainty
            if discrete
            else Metrics.symmetrical_unc_continuous_features
        )
        self._fitted = False

    @staticmethod
    def version() -> str:
        """Return the version of the package."""
        return __version__

    def _initialize(self, X, y):
        """Initialize the attributes so support multiple calls using same
        object

        Parameters
        ----------
        X : np.array
            array of features
        y : np.array
            vector of labels
        """
        self.X_ = X
        self.y_ = y
        if self.max_features is None:
            self._max_features = X.shape[1]
        else:
            self._max_features = self.max_features
        self._result = None
        self._scores = []
        self._su_labels = None
        self._su_features = {}
        self._fitted = True

    def _compute_su_labels(self):
        """Compute symmetrical uncertainty between each feature of the dataset
        and the labels and store it to use in future calls

        Returns
        -------
        list
            vector with sym. un. of every feature and the labels
        """
        if self._su_labels is None:
            num_features = self.X_.shape[1]
            self._su_labels = np.zeros(num_features)
            for col in range(num_features):
                self._su_labels[col] = self.symmetrical_uncertainty(
                    self.X_[:, col], self.y_
                )
        return self._su_labels

    def _compute_su_features(self, feature_a, feature_b):
        """Compute symmetrical uncertainty between two features and stores it
        to use in future calls

        Parameters
        ----------
        feature_a : int
            index of the first feature
        feature_b : int
            index of the second feature

        Returns
        -------
        float
            The symmetrical uncertainty of the two features
        """
        if (feature_a, feature_b) not in self._su_features:
            self._su_features[
                (feature_a, feature_b)
            ] = self.symmetrical_uncertainty_features(
                self.X_[:, feature_a], self.X_[:, feature_b]
            )
        return self._su_features[(feature_a, feature_b)]

    def _compute_merit(self, features):
        """Compute the merit function for cfs algorithms
           "Good feature subsets contain features highly correlated with
           (predictive of) the class, yet uncorrelated with (not predictive of)
           each other"
        Parameters
        ----------
        features : list
            list of features to include in the computation

        Returns
        -------
        float
            The merit of the feature set passed
        """
        # lgtm has already recognized that this is a false positive
        rcf = self._su_labels[
            features  # lgtm [py/hash-unhashable-value]
        ].sum()
        rff = 0.0
        k = len(features)
        for pair in list(combinations(features, 2)):
            rff += self._compute_su_features(*pair)
        return rcf / sqrt(k + (k**2 - k) * rff)

    def cfs(self, X, y):
        """Correlation-based Feature Selection
        with a forward best first heuristic search

        Parameters
        ----------
        X : np.array
            array of features
        y : np.array
            vector of labels

        Returns
        -------
        self
            self
        """
        self._initialize(X, y)
        s_list = self._compute_su_labels()
        # Descending order
        feature_order = (-s_list).argsort().tolist()
        continue_condition = True
        candidates = []
        # start with the best feature (max symmetrical uncertainty wrt label)
        first_candidate = feature_order.pop(0)
        candidates.append(first_candidate)
        self._scores.append(s_list[first_candidate])
        while continue_condition:
            merit = -float_info.min
            id_selected = None
            for idx, feature in enumerate(feature_order):
                candidates.append(feature)
                merit_new = self._compute_merit(candidates)
                if merit_new > merit:
                    id_selected = idx
                    merit = merit_new
                candidates.pop()
            candidates.append(feature_order[id_selected])
            self._scores.append(merit)
            del feature_order[id_selected]
            continue_condition = self._cfs_continue_condition(
                feature_order, candidates
            )
        self._result = candidates
        return self

    def _cfs_continue_condition(self, feature_order, candidates):
        if len(feature_order) == 0 or len(candidates) == self._max_features:
            # Force leaving the loop
            return False
        if len(self._scores) >= 5:
            """
            "To prevent the best first search from exploring the entire
            feature subset search space, a stopping criterion is imposed.
            The search will terminate if five consecutive fully expanded
            subsets show no improvement over the current best subset."
            as stated in Mark A. Hall Thesis
            """
            item_ant = -1
            for item in self._scores[-5:]:
                if item_ant == -1:
                    item_ant = item
                if item > item_ant:
                    break
                else:
                    item_ant = item
            else:
                return False
        return True

    def fcbf(self, X, y, threshold):
        """Fast Correlation-Based Filter

        Parameters
        ----------
        X : np.array
            array of features
        y : np.array
            vector of labels
        threshold : float
            threshold to select relevant features

        Returns
        -------
        self
            self

        Raises
        ------
        ValueError
            if the threshold is less than a selected value of 1e-7
        """
        if threshold < 1e-7:
            raise ValueError("Threshold cannot be less than 1e-7")
        self._initialize(X, y)
        s_list = self._compute_su_labels()
        feature_order = (-s_list).argsort()
        feature_dup = feature_order.copy().tolist()
        self._result = []
        for index_p in feature_order:
            # Don't self compare
            feature_dup.pop(0)
            # Remove redundant features
            if s_list[index_p] == 0.0:
                # the feature has been removed from the list
                continue
            if s_list[index_p] < threshold:
                break
            # Remove redundant features
            for index_q in feature_dup:
                su_pq = self._compute_su_features(index_p, index_q)
                if su_pq >= s_list[index_q]:
                    # remove feature from list
                    s_list[index_q] = 0.0
            self._result.append(index_p)
            self._scores.append(s_list[index_p])
            if len(self._result) == self._max_features:
                break
        return self

    def get_results(self):
        """Return the results of the algorithm applied if any

        Returns
        -------
        list
            list of features indices selected
        """
        return self._result if self._fitted else []

    def get_scores(self):
        """Return the scores computed for the features selected

        Returns
        -------
        list
            list of scores of the features selected
        """
        return self._scores if self._fitted else []

    def iwss(self, X, y, threshold):
        """Incremental Wrapper Subset Selection

        Parameters
        ----------
        X : np.array
            array of features
        y : np.array
            vector of labels
        threshold : float
            threshold to select relevant features

        Returns
        -------
        self
            self
        Raises
        ------
        ValueError
            if the threshold is less than a selected value of 1e-7
            or greater than .5

        """
        if threshold < 0 or threshold > 0.5:
            raise ValueError(
                "Threshold cannot be less than 0 or greater than 0.5"
            )
        self._initialize(X, y)
        s_list = self._compute_su_labels()
        feature_order = (-s_list).argsort()
        features = feature_order.copy().tolist()
        candidates = []
        # Add first and second features to result
        first_feature = features.pop(0)
        candidates.append(first_feature)
        self._scores.append(s_list[first_feature])
        candidates.append(features.pop(0))
        merit = self._compute_merit(candidates)
        self._scores.append(merit)
        for feature in features:
            candidates.append(feature)
            merit_new = self._compute_merit(candidates)
            delta = abs(merit - merit_new) / merit if merit != 0.0 else 0.0
            if merit_new > merit or delta < threshold:
                if merit_new > merit:
                    merit = merit_new
                self._scores.append(merit_new)
            else:
                candidates.pop()
                break
            if len(candidates) == self._max_features:
                break
        self._result = candidates
        return self
