""" Non-negative matrix factorization for I divergence

    This code was implements Lee and Seung's multiplicative updates algorithm
    for NMF with I divergence cost.

    Lee D. D., Seung H. S., Learning the parts of objects by non-negative
      matrix factorization. Nature, 1999
"""
# Author: Olivier Mangin <olivier.mangin@inria.fr>


import warnings

import numpy as np
import scipy.sparse as sp
from math import sqrt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
from sklearn.utils.extmath import safe_sparse_dot

def norm(x):
    """Dot product-based Euclidean norm implementation
    See: http://fseoane.net/blog/2011/computing-the-vector-norm/
    """
    return sqrt(squared_norm(x))

def check_non_negative(X, whom):
    X = X.data if sp.issparse(X) else X
    if (X < 0).any():
        raise ValueError("Negative values in data passed to %s" % whom)


def _normalize_sum(a, axis=0, eps=1.e-16):
    if axis >= len(a.shape):
        raise ValueError
    return a / (eps + np.expand_dims(np.sum(a, axis=axis), axis))


def _scale(matrix, factors, axis=0):
    """Scales line or columns of a matrix.

    Parameters
    ----------
    :param matrix: 2-dimensional array
    :param factors: 1-dimensional array
    :param axis: 0: columns are scaled, 1: lines are scaled
    """
    if not (len(matrix.shape) == 2):
        raise ValueError(
                "Wrong array shape: %s, should have only 2 dimensions."
                % str(matrix.shape))
    if not axis in (0, 1):
        raise ValueError('Wrong axis, should be 0 (scaling lines)\
                or 1 (scaling columns).')
    # Transform factors given as columne shaped matrices
    factors = np.squeeze(np.asarray(factors))
    if axis == 1:
        factors = factors[:, np.newaxis]
    return np.multiply(matrix, factors)


def generalized_KL(x, y, eps=1.e-8, axis=None):
    return (np.multiply(x, np.log(np.divide(x + eps, y + eps))) - x + y
            ).sum(axis=axis)


def _special_sparse_dot(a, b, refmat):
    """Computes dot product of a and b on indices where refmat is nonnzero
    and returns sparse csr matrix with same structure than refmat.

    First calls to eliminate_zeros on refmat which might modify the structure
    of refmat.

    Params
    ------
    a, b: dense arrays
    refmat: sparse matrix

    Dot product of a and b must have refmat's shape.
    """
    refmat.eliminate_zeros()
    ii, jj = refmat.nonzero()
    dot_vals = np.multiply(a[ii, :], b.T[jj, :]).sum(axis=1)
    c = sp.coo_matrix((dot_vals, (ii, jj)), shape=refmat.shape)
    return c.tocsr()


class KLdivNMF(BaseEstimator, TransformerMixin):
    """Non negative factorization with Kullback Leibler divergence cost.

    Parameters
    ----------
    n_components: int or None
        Number of components, if n_components is not set all components
        are kept

    init:  'nndsvd' |  'nndsvda' | 'nndsvdar' | 'random'
        Method used to initialize the procedure.
        Default: 'nndsvdar' if n_components < n_features, otherwise random.
        Valid options::

            'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
                initialization (better for sparseness)
            'nndsvda': NNDSVD with zeros filled with the average of X
                (better when sparsity is not desired)
            'nndsvdar': NNDSVD with zeros filled with small random values
                (generally faster, less accurate alternative to NNDSVDa
                for when sparsity is not desired)
            'random': non-negative random matrices

    tol: double, default: 1e-4
        Tolerance value used in stopping conditions.

    max_iter: int, default: 200
        Number of iterations to compute.

    subit: int, default: 10
        Number of sub-iterations to perform on W (resp. H) before switching
        to H (resp. W) update.

    Attributes
    ----------
    `components_` : array, [n_components, n_features]
        Non-negative components of the data

    random_state : int or RandomState
        Random number generator seed control.

    Examples
    --------

    >>> import numpy as np
    >>> X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> from kl_nmf import KLdivNMF
    >>> model = KLdivNMF(n_components=2, init='random', random_state=0)
    >>> model.fit(X) #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    KLdivNMF(eps=1e-08, init='random', max_iter=200, n_components=2,
            random_state=0, subit=10, tol=1e-06)
    >>> model.components_
    array([[ 0.50303234,  0.49696766],
           [ 0.93326505,  0.06673495]])

    Notes
    -----
    This implements

    Lee D. D., Seung H. S., Learning the parts of objects by non-negative
      matrix factorization. Nature, 1999
    """

    def __init__(self, n_components=None, tol=1e-6, max_iter=200, eps=1.e-8,
                 subit=10, random_state=None, init=None):
        self.n_components = n_components
        self._init_dictionary = None
        self.random_state = random_state
        self.tol = tol
        self.max_iter = max_iter
        self.eps = eps
        self.init = init
        # Only for gradient updates
        self.subit = subit


    def _initialize_nmf(self, X, n_components, init=None, eps=1e-6,
                    random_state=None):
        """
        Algorithms for NMF initialization.
        Computes an initial guess for the non-negative
        rank k matrix approximation for X: X = WH
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data matrix to be decomposed.
        n_components : integer
            The number of components desired in the approximation.
        init :  None | 'random' | 'nndsvd' | 'nndsvda' | 'nndsvdar'
            Method used to initialize the procedure.
            Default: 'nndsvdar' if n_components < n_features, otherwise 'random'.
            Valid options:
            - 'random': non-negative random matrices, scaled with:
                sqrt(X.mean() / n_components)
            - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
                initialization (better for sparseness)
            - 'nndsvda': NNDSVD with zeros filled with the average of X
                (better when sparsity is not desired)
            - 'nndsvdar': NNDSVD with zeros filled with small random values
                (generally faster, less accurate alternative to NNDSVDa
                for when sparsity is not desired)
            - 'custom': use custom matrices W and H
        eps : float
            Truncate all values less then this in output to zero.
        random_state : int seed, RandomState instance, or None (default)
            Random number generator seed control, used in 'nndsvdar' and
            'random' modes.
        Returns
        -------
        W : array-like, shape (n_samples, n_components)
            Initial guesses for solving X ~= WH
        H : array-like, shape (n_components, n_features)
            Initial guesses for solving X ~= WH
        References
        ----------
        C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
        nonnegative matrix factorization - Pattern Recognition, 2008
        http://tinyurl.com/nndsvd
        """

        check_non_negative(X, "NMF initialization")
        n_samples, n_features = X.shape

        if init is None:
            if n_components < n_features:
                init = 'nndsvd'
            else:
                init = 'random'

        # Random initialization
        if init == 'random':
            avg = np.sqrt(X.mean() / n_components)
            rng = check_random_state(random_state)
            H = avg * rng.randn(n_components, n_features)
            W = avg * rng.randn(n_samples, n_components)
            # we do not write np.abs(H, out=H) to stay compatible with
            # numpy 1.5 and earlier where the 'out' keyword is not
            # supported as a kwarg on ufuncs
            np.abs(H, H)
            np.abs(W, W)
            return W, H

        # NNDSVD initialization
        U, S, V = randomized_svd(X, n_components, random_state=random_state)
        W, H = np.zeros(U.shape), np.zeros(V.shape)

        # The leading singular triplet is non-negative
        # so it can be used as is for initialization.
        W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
        H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

        for j in range(1, n_components):
            x, y = U[:, j], V[j, :]

            # extract positive and negative parts of column vectors
            x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
            x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

            # and their norms
            x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
            x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

            m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

            # choose update
            if m_p > m_n:
                u = x_p / x_p_nrm
                v = y_p / y_p_nrm
                sigma = m_p
            else:
                u = x_n / x_n_nrm
                v = y_n / y_n_nrm
                sigma = m_n

            lbd = np.sqrt(S[j] * sigma)
            W[:, j] = lbd * u
            H[j, :] = lbd * v

        W[W < eps] = 0
        H[H < eps] = 0

        if init == "nndsvd":
            pass
        elif init == "nndsvda":
            avg = X.mean()
            W[W == 0] = avg
            H[H == 0] = avg
        elif init == "nndsvdar":
            rng = check_random_state(random_state)
            avg = X.mean()
            W[W == 0] = abs(avg * rng.randn(len(W[W == 0])) / 100)
            H[H == 0] = abs(avg * rng.randn(len(H[H == 0])) / 100)
        else:
            raise ValueError(
                'Invalid init parameter: got %r instead of one of %r' %
                (init, (None, 'random', 'nndsvd', 'nndsvda', 'nndsvdar')))

        return W, H

    def fit_transform(self, X, y=None, weights=1., _fit=True,
                      return_errors=False, scale_W=False):
        """Learn a NMF model for the data X and returns the transformed data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------

        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Data matrix to be decomposed

        weights: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Weights on the cost function used as coefficients on each
            element of the data. If smaller dimension is provided, standard
            numpy broadcasting is used.

        return_errors: boolean
            if True, the list of reconstruction errors along iterations is
            returned

        scale_W: boolean (default: False)
            Whether to force scaling of W during updates. This is only relevant
            if components are normalized.

        _fit: if True (default), update the model, else only compute transform

        Returns
        -------
        data: array, [n_samples, n_components]
            Transformed data

        or (data, errors) if return_errors
        """
        X = check_array(X, accept_sparse=('csr', 'csc'))
        check_non_negative(X, "NMF.fit")

        n_samples, n_features = X.shape

        if not self.n_components:
            self.n_components = n_features

        W, H = self._initialize_nmf(X, self.n_components, init=self.init,
                                random_state=self.random_state)

        if _fit:
            self.components_ = H

        prev_error = np.Inf
        tol = self.tol * n_samples * n_features

        if return_errors:
            errors = []

        for n_iter in xrange(1, self.max_iter + 1):
            # Stopping condition
            error = self.error(X, W, self.components_, weights=weights)
            if prev_error - error < tol:
                break
            prev_error = error

            if return_errors:
                errors.append(error)
            W = self._update(X, W, _fit=_fit)

        if n_iter == self.max_iter and tol > 0:
            warnings.warn("Iteration limit reached during fit\n")

        if return_errors:
            return W, errors
        else:
            return W

    def _update(self, X, W, _fit=True, scale_W=False, eps=1.e-8):
        """Perform one update iteration.

        Updates components if _fit and returns updated coefficients.

        Params:
        -------
            _fit: boolean (default: True)
                Whether to update components.

            scale_W: boolean (default: False)
                Whether to force scaling of W. This is only relevant if
                components are normalized.
        """
        if scale_W:
            # This is only relevant if components are normalized.
            # Not always usefull but might improve convergence speed:
            # Scale W lines to have same sum than X lines
            W = _scale(_normalize_sum(W, axis=1), X.sum(axis=1), axis=1)
        Q = self._Q(X, W, self.components_, eps=eps)
        # update W
        W = self._updated_W(X, W, self.components_, Q=Q)
        if _fit:
            # update H
            self.components_ = self._updated_H(X, W, self.components_, Q=Q)
        return W

    def fit(self, X, y=None, **params):
        """Learn a NMF model for the data X.

        Parameters
        ----------

        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Data matrix to be decomposed

        Returns
        -------
        self
        """
        self.fit_transform(X, **params)
        return self

    def transform(self, X, **params):
        """Transform the data X according to the fitted NMF model

        Parameters
        ----------

        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Data matrix to be transformed by the model

        Returns
        -------
        data: array, [n_samples, n_components]
            Transformed data
        """
        self._init_dictionary = self.components_
        params['_fit'] = False
        return self.fit_transform(X, **params)

    # Helpers for beta divergence and related updates

    # Errors and performance estimations

    def error(self, X, W, H=None, weights=1., eps=1.e-8):
        X = check_array(X, accept_sparse='csr')
        if H is None:
            H = self.components_
        if sp.issparse(X):
            WH = _special_sparse_dot(W, H, X)
            # Avoid computing all values of WH to get their sum
            WH_sum = np.sum(np.multiply(np.sum(W, axis=0), np.sum(H, axis=1)))
            return (np.multiply(
                X.data,
                np.log(np.divide(X.data + eps, WH.data + eps))
                )).sum() - X.data.sum() + WH_sum
        else:
            return generalized_KL(X, np.dot(W, H))

    # Projections

    def scale(self, W, H, factors):
        """Scale W columns and H rows inversely, according to the given
        coefficients.
        """
        safe_factors = factors + self.eps
        s_W = _scale(W, safe_factors, axis=0)
        s_H = _scale(H, 1. / safe_factors, axis=1)
        return s_W, s_H

    # Update rules

    @classmethod
    def _Q(cls, X, W, H, eps=1.e-8):
        """Computes X / (WH)
           where '/' is element-wise and WH is a matrix product.
        """
        # X should be at least 2D or csr
        if sp.issparse(X):
            WH = _special_sparse_dot(W, H, X)
            WH.data = (X.data + eps) / (WH.data + eps)
            return WH
        else:
            return np.divide(X + eps, np.dot(W, H) + eps)

    @classmethod
    def _updated_W(cls, X, W, H, weights=1., Q=None, eps=1.e-8):
        if Q is None:
            Q = cls._Q(X, W, H, eps=eps)
        W = np.multiply(W, safe_sparse_dot(Q, H.T))
        return W

    @classmethod
    def _updated_H(cls, X, W, H, weights=1., Q=None, eps=1.e-8):
        if Q is None:
            Q = cls._Q(X, W, H, eps=eps)
        H = np.multiply(H, safe_sparse_dot(W.T, Q))
        H = _normalize_sum(H, axis=1)
        return H