from __future__ import annotations
import abc
import inspect
import math
import numpy as np

from ohkubolib.datamodel import datamodel, field

# 辞書を定義するための抽象クラス
class Dictionary(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def __call__(self, X: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def diff(self, X: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def ddiff(self, X: np.ndarray) -> np.ndarray:
        pass


class StateDictionary(Dictionary, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def s2i(self, state: list[int]) -> int:
        pass

    @abc.abstractmethod
    def i2s(self, id: int) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def num_states(self) -> int:
        pass


# # 単項式辞書のクラス
class Monomial(Dictionary):
    order: int                # 最大次数
    powers: np.ndarray | None # 単項式の冪乗を計算するための numpy 配列 (初期値: None)
    def __init__(self, order: int) -> None:
        if type(order) is not int:
            raise ValueError(f"'order' in '{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}()' is expected type '<class 'int'>', but type '{type(order)}'.")

        self.order = order
        self.powers = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x[:, np.newaxis]

        dim, M = x.shape
        if self.powers is None:
            self.powers = self.make_powers(dim, self.order)
        N_K: int = self.powers.shape[1]
        y: np.ndarray = np.ones([N_K, M])

        for i in range(N_K):
            for j in range(dim):
                y[i, :] = y[i, :] * np.power(x[j, :], self.powers[j, i])

        return y

    def diff(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x[:, np.newaxis]

        dim, M = x.shape
        if self.powers is None:
            self.powers = self.make_powers(dim, self.order)
        N_K: int = self.powers.shape[1]
        y: np.ndarray = np.zeros([N_K, dim, M])

        for i in range(N_K):
            for j in range(dim):
                e: np.ndarray = self.powers[:, i].copy()
                a: np.ndarray = e[j]
                e[j] = e[j] - 1

                if np.any(e < 0):
                    continue

                y[i, j, :] = a*np.ones([1, M])
                for k in range(dim):
                    y[i, j, :] = y[i, j, :] * np.power(x[k, :], e[k])

        return y

    def ddiff(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x[:, np.newaxis]

        dim, M = x.shape
        if self.powers is None:
            self.powers = self.make_powers(dim, self.order)
        N_K: int = self.powers.shape[1]
        y: np.ndarray = np.ones([N_K, dim, dim, M])

        for i in range(N_K):
            for j1 in range(dim):
                for j2 in range(dim):
                    e: np.ndarray = self.powers[:, i].copy()
                    a: np.ndarray = e[j1]
                    e[j1] = e[j1] - 1
                    a += e[j2]
                    e[j2] = e[j2] - 1

                    if np.any(e < 0):
                        continue

                    y[i, j1, j2, :] = a*np.ones([1, M])
                    for k in range(dim):
                        y[i, j1, j2, :] = y[i, j1, j2, :] * np.power(x[k, :], e[k])

        return y

    @staticmethod
    def make_powers(dim: int, order: int) -> np.ndarray:
        def nchoosek(N_K: int, k: int) -> int:
            return math.factorial(N_K)//math.factorial(k)//math.factorial(N_K-k)

        def next_powers(x: np.ndarray) -> np.ndarray:
            M = len(x)
            j = 0
            for i in range(1, M):
                if x[i] > 0:
                    j = i
                    break

            if j == 0:
                t = x[0]
                x[0] = 0
                x[M-1] = t + 1
            elif j < M - 1:
                x[j] = x[j] - 1
                t = x[0] + 1
                x[0] = 0
                x[j-1] = x[j-1] + t
            elif j == M - 1:
                t = x[0]
                x[0] = 0
                x[j-1] = t + 1
                x[j] = x[j] - 1

            return x

        M: int = nchoosek(order+dim, order)
        x: np.ndarray = np.zeros(dim)
        c: np.ndarray = np.zeros([dim, M])
        for i in range(1, M):
            c[:, i] = next_powers(x)
        c = np.flipud(c)

        return c

@datamodel
class KronMonomial(StateDictionary):
    """
    Set a monomial dictionary.

    Parameters for initialization
    ----------
    dim : int
        dimension of the input vector.

    order : int
        maximum order of monomials.
    """
    dim: int
    order: int
    powers: np.ndarray = field(init=False, ignore_serde=True)

    def __post_init__(self) -> None:
        original_comp = np.arange(self.order+1)
        comp = np.copy(original_comp.reshape(self.order+1, 1))
        for d in range(1, self.dim):
            comp2 = np.repeat(comp, self.order+1, axis=0)
            comp3 = np.tile(original_comp,int(comp2.shape[0]/(self.order+1)))
            comp3 = comp3.reshape(comp3.shape[0],1)
            comp = np.hstack((comp3,comp2))
        self.powers = comp.T


    def __call__(self, X: np.ndarray, matrix_style: str = 'dim-first') -> np.ndarray:
        if matrix_style == 'dim-first':
            X = X.T

        if X.ndim == 1:
            X = X[np.newaxis, :]

        num_data, dim = X.shape
        num_dic: int = self.powers.shape[1]
        psiX = np.zeros([num_dic, num_data])

        for i in range(num_data):
            psiX[:, i] = np.prod(np.power(X[i, :], self.powers.T), axis=1)

        return psiX


    def diff(self, X: np.ndarray) -> np.ndarray:
        return np.empty(0)

    def ddiff(self, X: np.ndarray) -> np.ndarray:
        return np.empty(0)

    def s2i(self, state: list[int]) -> int:
        '''
        Convert a given index vector to the corresponding scalar index number in the monomial disctionary

        Example: dim=2, max_order=4
        [0 0] -> 0
        [1 0] -> 1
        ...
        [4 0] -> 4
        [0 1] -> 5

        Parameters
        ----------
        index_vec: np.array
            shape(d), a state vector indicating the index in the monomial dictionary
        '''
        comp = 0
        for d in range(self.dim):
            comp = comp + state[d]*np.power(self.order+1, d)
        return int(comp)


    def i2s(self, id: int) -> np.ndarray:
        """
        Convert a scalar index number to the corresponding index vector in the monomial dictionary.

        Parameters
        ----------
        id: int
            A scalar index number indicating the index in the monomial dictionary.
        """
        return self.powers[:,id]

    @property
    def num_states(self) -> int:
        """
        Get the number of states in the monomial dictionary.

        Returns
        -------
        int
            The number of states in the monomial dictionary.
        """
        return self.powers.shape[1]
