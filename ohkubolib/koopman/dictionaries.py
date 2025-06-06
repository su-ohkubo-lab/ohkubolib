from __future__ import annotations
import abc
import inspect
import math
import numpy as np

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