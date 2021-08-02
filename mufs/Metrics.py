from math import log
import numpy as np

from scipy.special import digamma, gamma, psi
from sklearn.neighbors import BallTree, KDTree
from sklearn.neighbors import NearestNeighbors


class Metrics:
    @staticmethod
    def information_gain_cont(x, y):
        """Measures the reduction in uncertainty about the value of y when the
        value of X continuous is known (also called mutual information)
        (https://www.sciencedirect.com/science/article/pii/S0020025519303603)

        Parameters
        ----------
        x : np.array
            values of the continuous variable
        y : np.array
            array of labels

        Returns
        -------
        float
            Information gained
        """
        return Metrics._compute_mi_cd(x, y, n_neighbors=3)

    @staticmethod
    def information_gain_cont_features(xa, xb):
        """Measures the reduction in uncertainty about the value of xb when the
        value of xa continuous is known (also called mutual information)
        (https://www.sciencedirect.com/science/article/pii/S0020025519303603)

        Parameters
        ----------
        xa : np.array
            values of the continuous variable
        xb : np.array
            values of the continuous variable

        Returns
        -------
        float
            Information gained
        """
        return Metrics._compute_mi_cc(xa, xb, n_neighbors=3)

    @staticmethod
    def _compute_mi_cc(x, y, n_neighbors):
        """Compute mutual information between two continuous variables.

        # Author: Nikolay Mayorov <n59_ru@hotmail.com>
        # License: 3-clause BSD

        Parameters
        ----------
        x, y : ndarray, shape (n_samples,)
            Samples of two continuous random variables, must have an identical
            shape.

        n_neighbors : int
            Number of nearest neighbors to search for each point, see [1]_.

        Returns
        -------
        mi : float
            Estimated mutual information. If it turned out to be negative it is
            replace by 0.

        Notes
        -----
        True mutual information can't be negative. If its estimate by a
        numerical method is negative, it means (providing the method is
        adequate) that the mutual information is close to 0 and replacing it by
        0 is a reasonable strategy.

        References
        ----------
        .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
            information". Phys. Rev. E 69, 2004.
        """

        n_samples = x.size

        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))
        xy = np.hstack((x, y))

        # Here we rely on NearestNeighbors to select the fastest algorithm.
        nn = NearestNeighbors(metric="chebyshev", n_neighbors=n_neighbors)

        nn.fit(xy)
        radius = nn.kneighbors()[0]
        radius = np.nextafter(radius[:, -1], 0)

        # KDTree is explicitly fit to allow for the querying of number of
        # neighbors within a specified radius
        kd = KDTree(x, metric="chebyshev")
        nx = kd.query_radius(x, radius, count_only=True, return_distance=False)
        nx = np.array(nx) - 1.0

        kd = KDTree(y, metric="chebyshev")
        ny = kd.query_radius(y, radius, count_only=True, return_distance=False)
        ny = np.array(ny) - 1.0

        mi = (
            digamma(n_samples)
            + digamma(n_neighbors)
            - np.mean(digamma(nx + 1))
            - np.mean(digamma(ny + 1))
        )

        return max(0, mi)

    @staticmethod
    def _compute_mi_cd(c, d, n_neighbors):
        """Compute mutual information between continuous and discrete
        variable.

        # Author: Nikolay Mayorov <n59_ru@hotmail.com>
        # License: 3-clause BSD


        Parameters
        ----------
        c : ndarray, shape (n_samples,)
            Samples of a continuous random variable.

        d : ndarray, shape (n_samples,)
            Samples of a discrete random variable.

        n_neighbors : int
            Number of nearest neighbors to search for each point, see [1]_.

        Returns
        -------
        mi : float
            Estimated mutual information. If it turned out to be negative it is
            replace by 0.

        Notes
        -----
        True mutual information can't be negative. If its estimate by a
        numerical method is negative, it means (providing the method is
        adequate) that the mutual information is close to 0 and replacing it
        by 0 is a reasonable strategy.

        References
        ----------
        .. [1] B. C. Ross "Mutual Information between Discrete and Continuous
        Data Sets". PLoS ONE 9(2), 2014.
        """
        n_samples = c.shape[0]
        if c.ndim == 1:
            c = c.reshape((-1, 1))
        radius = np.empty(n_samples)
        label_counts = np.empty(n_samples)
        k_all = np.empty(n_samples)
        nn = NearestNeighbors()
        for label in np.unique(d):
            mask = d == label
            count = np.sum(mask)
            if count > 1:
                k = min(n_neighbors, count - 1)
                nn.set_params(n_neighbors=k)
                nn.fit(c[mask])
                r = nn.kneighbors()[0]
                radius[mask] = np.nextafter(r[:, -1], 0)
                k_all[mask] = k
            label_counts[mask] = count
        # Ignore points with unique labels.
        mask = label_counts > 1
        n_samples = np.sum(mask)
        label_counts = label_counts[mask]
        k_all = k_all[mask]
        c = c[mask]
        radius = radius[mask]
        if n_samples == 0:
            return 0.0
        kd = (
            BallTree(c, metric="chebyshev")
            if n_samples >= 20
            else KDTree(c, metric="chebyshev")
        )
        m_all = kd.query_radius(
            c, radius, count_only=True, return_distance=False
        )
        m_all = np.array(m_all) - 1.0
        mi = (
            digamma(n_samples)
            + np.mean(digamma(k_all))
            - np.mean(digamma(label_counts))
            - np.mean(digamma(m_all + 1))
        )
        return max(0.0, mi)

    @staticmethod
    def _nearest_distances(X, k=1):
        """
        X = array(N,M)
        N = number of points
        M = number of dimensions
        returns the distance to the kth nearest neighbor for every point in X
        """
        knn = NearestNeighbors(n_neighbors=k + 1)
        knn.fit(X)
        d, _ = knn.kneighbors(X)  # the first nearest neighbor is itself
        return d[:, -1]  # returns the distance to the kth nearest neighbor

    @staticmethod
    def differential_entropy(x, k=1):
        """Returns the entropy of the X.
        Parameters
        ===========
        x : array-like, shape (n_samples, n_features)
            The data the entropy of which is computed
        k : int, optional
            number of nearest neighbors for density estimation
        Notes
        ======
        Kozachenko, L. F. & Leonenko, N. N. 1987 Sample estimate of entropy
        of a random vector. Probl. Inf. Transm. 23, 95-101.
        See also: Evans, D. 2008 A computationally efficient estimator for
        mutual information, Proc. R. Soc. A 464 (2093), 1203-1215.
        and:
        Kraskov A, Stogbauer H, Grassberger P. (2004). Estimating mutual
        information. Phys Rev E 69(6 Pt 2):066138.

        Differential entropy can be negative
        https://stats.stackexchange.com/questions/73881/
        when-is-the-differential-entropy-negative
        """
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        # Distance to kth nearest neighbor
        r = Metrics._nearest_distances(x, k)  # squared distances
        n, d = x.shape
        volume_unit_ball = (np.pi ** (0.5 * d)) / gamma(0.5 * d + 1)
        """
        F. Perez-Cruz, (2008). Estimation of Information Theoretic Measures
        for Continuous Random Variables. Advances in Neural Information
        Processing Systems 21 (NIPS). Vancouver (Canada), December.
        return d*mean(log(r))+log(volume_unit_ball)+log(n-1)-log(k)
        """
        return (
            d * np.mean(np.log(r + np.finfo(x.dtype).eps))
            + np.log(volume_unit_ball)
            + psi(n)
            - psi(k)
        )

    @staticmethod
    def symmetrical_unc_continuous(x, y):
        """Compute symmetrical uncertainty. Using Greg Ver Steeg's npeet
        https://github.com/gregversteeg/NPEET

        Parameters
        ----------
        x : np.array
            values of the continuous variable
        y : np.array
            array of labels

        Returns
        -------
        float
            symmetrical uncertainty
        """

        return (
            2.0
            * Metrics.information_gain_cont(x, y)
            / (
                Metrics.differential_entropy(x, k=len(x) - 1)
                + Metrics.entropy(y)
            )
        )

    @staticmethod
    def symmetrical_unc_continuous_features(x, y):
        """Compute symmetrical uncertainty. Using Greg Ver Steeg's npeet
        https://github.com/gregversteeg/NPEET

        Parameters
        ----------
        x : np.array
            values of the continuous variable
        y : np.array
            array of labels

        Returns
        -------
        float
            symmetrical uncertainty
        """

        return (
            2.0
            * Metrics.information_gain_cont_features(x, y)
            / (
                Metrics.differential_entropy(x, k=len(x) - 1)
                + Metrics.entropy(y)
            )
        )

    @staticmethod
    def symmetrical_uncertainty(x, y):
        """Compute symmetrical uncertainty. Normalize* information gain (mutual
        information) with the entropies of the features in order to compensate
        the bias due to high cardinality features. *Range [0, 1]
        (https://www.sciencedirect.com/science/article/pii/S0020025519303603)

        Parameters
        ----------
        x : np.array
            values of the variable
        y : np.array
            array of labels

        Returns
        -------
        float
            symmetrical uncertainty
        """
        return (
            2.0
            * Metrics.information_gain(x, y)
            / (Metrics.entropy(x) + Metrics.entropy(y))
        )

    @staticmethod
    def conditional_entropy(x, y, base=2):
        """quantifies the amount of information needed to describe the outcome
        of Y given that the value of X is known
        computes H(Y|X)

        Parameters
        ----------
        x : np.array
            values of the variable
        y : np.array
            array of labels
        base : int, optional
            base of the logarithm, by default 2

        Returns
        -------
        float
            conditional entropy of y given x
        """
        xy = np.c_[x, y]
        return Metrics.entropy(xy, base) - Metrics.entropy(x, base)

    @staticmethod
    def entropy(y, base=2):
        """measure of the uncertainty in predicting the value of y

        Parameters
        ----------
        y : np.array
            array of labels
        base : int, optional
            base of the logarithm, by default 2

        Returns
        -------
        float
            entropy of y
        """
        _, count = np.unique(y, return_counts=True, axis=0)
        proba = count.astype(float) / len(y)
        proba = proba[proba > 0.0]
        return np.sum(proba * np.log(1.0 / proba)) / log(base)

    @staticmethod
    def information_gain(x, y, base=2):
        """Measures the reduction in uncertainty about the value of y when the
        value of X is known (also called mutual information)
        (https://www.sciencedirect.com/science/article/pii/S0020025519303603)

        Parameters
        ----------
        x : np.array
            values of the variable
        y : np.array
            array of labels
        base : int, optional
            base of the logarithm, by default 2

        Returns
        -------
        float
            Information gained
        """
        return Metrics.entropy(y, base) - Metrics.conditional_entropy(
            x, y, base
        )
