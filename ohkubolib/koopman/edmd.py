from __future__ import annotations
from typing import Literal
import inspect
import pathlib
import numpy as np
import scipy.linalg as LA
from sklearn.linear_model import Lasso, Ridge
from ohkubolib.koopman.dictionaries import Dictionary

# EDMD のクラス
class EDMD(object):
    dictionary_X: Dictionary                  # x ベクトルに対する辞書
    dictionary_Y: Dictionary                  # y ベクトルに対する辞書 (dictionary で定義した場合, x ベクトルに対する辞書と同じ)
    mode: Literal['normal', 'lasso', 'ridge'] # normal: 通常の EDMD, lasso: Lasso ありの EDMD, ridge: Ridge ありの EDMD (初期値: normal)
    alpha: float                              # Lasso (or Ridge) の正則化係数 (初期値: 1.0)
    iterations: int                           # Lasso (or Ridge) の反復回数 (初期値: 1000)
    K: np.ndarray | None                      # Koopman 行列 (初期値: None)
    def __init__(self, dictionary: Dictionary | None = None, dictionary_X: Dictionary | None = None, dictionary_Y: Dictionary | None = None , mode: Literal['normal', 'lasso', 'ridge'] = 'normal', alpha: int | float = 1.0, iterations: int = 1000) -> None:
        super(EDMD, self).__init__()

        if not isinstance(dictionary, Dictionary) and dictionary is not None:
            raise ValueError(f"'dictionary' in '{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}()' is expected type '<class 'Dictionary'>' or 'None', but type '{type(dictionary)}'.")

        if not isinstance(dictionary_X, Dictionary) and dictionary_X is not None:
            raise ValueError(f"'dictionary_X' in '{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}()' is expected type '<class 'Dictionary'>' or 'None', but type '{type(dictionary_X)}'.")

        if not isinstance(dictionary_Y, Dictionary) and dictionary_Y is not None:
            raise ValueError(f"'dictionary_Y' in '{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}()' is expected type '<class 'Dictionary'>' or 'None', but type '{type(dictionary_Y)}'.")

        if not (isinstance(dictionary, Dictionary) and dictionary_X is None and dictionary_Y is None) \
            and not (dictionary is None and isinstance(dictionary_X, Dictionary) and isinstance(dictionary_Y, Dictionary)):
            raise ValueError(f"Using 'dictionary' or 'dictionary_X and dictionary_Y' in '{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}()'. The other is expected 'None'.")

        if mode not in ['normal', 'lasso', 'ridge']:
            raise ValueError(f"'mode' in '{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}()' is expected 'normal' or 'lasso' or 'ridge', but '{mode}'.")

        if type(alpha) is not int and type(alpha) is not float:
            raise ValueError(f"'alpha' in '{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}()' is expected type '<class 'int'>' or '<class 'float'>', but type '{type(alpha)}'.")

        if type(iterations) is not int:
            raise ValueError(f"'iterations' in '{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}()' is expected type '<class 'int'>', but type '{type(iterations)}'.")

        if dictionary is None:
            self.dictionary_X = dictionary_X
            self.dictionary_Y = dictionary_Y
        else:
            self.dictionary_X = dictionary
            self.dictionary_Y = dictionary
        self.mode = mode
        self.alpha = alpha
        self.iterations = iterations
        self.K = None

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        PsiX: np.ndarray = self.dictionary_X(X)
        PsiY: np.ndarray = self.dictionary_Y(Y)

        if self.mode == 'normal':
            G: np.ndarray = PsiX @ PsiX.T
            A: np.ndarray = PsiX @ PsiY.T
            self.K = LA.pinv(G) @ A

        elif self.mode == 'lasso':
            clf: Ridge = Lasso(alpha=self.alpha, fit_intercept=False, max_iter=self.iterations).fit(PsiX.T, PsiY.T)
            self.K = clf.coef_.T

        elif self.mode == 'ridge':
            clf: Ridge = Ridge(alpha=self.alpha, fit_intercept=False, max_iter=self.iterations).fit(PsiX.T, PsiY.T)
            self.K = clf.coef_.T

        return self.K

    def save(self, path: str) -> None:
        if self.K is None:
            print(f"'K' in '{self.__class__.__name__}' is no data.")
        else:
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(path, X=self.K)


# Online EDMD のクラス
class OnlineEDMD(object):
    K: np.ndarray | None
    def __init__(self) -> None:
        super(OnlineEDMD, self).__init__()
        self.K = None

    def __call__(self) -> np.ndarray:
        pass

    def save(self, path: str) -> None:
        if self.K is None:
            print(f"'K' in '{self.__class__.__name__}' is no data.")
        else:
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(path, X=self.K)