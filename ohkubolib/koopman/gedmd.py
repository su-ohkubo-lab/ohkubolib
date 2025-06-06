from __future__ import annotations
from typing import Literal
import inspect
import pathlib
import numpy as np
import scipy.linalg as LA
from sklearn.linear_model import Lasso, Ridge
from ohkubolib.koopman.dictionaries import Dictionary

# gEDMD のクラス
class gEDMD(object):
    dictionary: Dictionary                    # x ベクトル & y ベクトルに対する辞書
    mode: Literal['normal', 'lasso', 'ridge'] # normal: 通常の gEDMD, lasso: Lasso ありの gEDMD, ridge: Ridge ありの gEDMD (初期値: normal)
    alpha: float                              # Lasso (or Ridge) の正則化係数 (初期値: 1.0)
    iterations: int                           # Lasso (or Ridge) の反復回数 (初期値: 1000)
    L: np.ndarray | None                      # Koopman 生成子行列 (初期値: None)
    def __init__(self, dictionary: Dictionary, mode: Literal['normal', 'lasso', 'ridge'] = 'normal', alpha: int | float = 1.0, iterations: int = 1000) -> None:
        super(gEDMD, self).__init__()

        if not isinstance(dictionary, Dictionary):
            raise ValueError(f"'dictionary' in '{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}()' is expected type '<class 'Dictionary'>', but type '{type(dictionary)}'.")

        if mode not in ['normal', 'lasso', 'ridge']:
            raise ValueError(f"'mode' in '{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}()' is expected 'normal' or 'lasso' or 'ridge', but '{mode}'.")

        if type(alpha) is not int and type(alpha) is not float:
            raise ValueError(f"'alpha' in '{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}()' is expected type '<class 'int'>' or '<class 'float'>', but type '{type(alpha)}'.")

        if type(iterations) is not int:
            raise ValueError(f"'iterations' in '{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}()' is expected type '<class 'int'>', but type '{type(iterations)}'.")
    
        self.dictionary = dictionary
        self.mode = mode
        self.alpha = alpha
        self.iterations = iterations
        self.L = None

    def __call__(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        PsiX: np.ndarray = self.dictionary(X)
        dPsiY: np.ndarray = np.einsum('ijk,jk->ik', self.dictionary.diff(X), Y)
        if Z is not None:
            ddPsiX: np.ndarray = self.dictionary.ddiff(X)
            for i in range(PsiX.shape[0]):
                dPsiY[i, :] += 0.5 * np.sum(ddPsiX[i, :, :, :]*Z, axis=(0, 1))

        if self.mode == 'normal':
            C_0: np.ndarray = PsiX @ PsiX.T
            C_1: np.ndarray = PsiX @ dPsiY.T
            self.L = LA.pinv(C_0) @ C_1

        elif self.mode == 'lasso':
            clf: Lasso = Lasso(alpha=self.alpha, fit_intercept=False, max_iter=self.iterations).fit(PsiX.T, dPsiY.T)
            self.L = clf.coef_.T

        elif self.mode == 'ridge':
            clf: Ridge = Ridge(alpha=self.alpha, fit_intercept=False, max_iter=self.iterations).fit(PsiX.T, dPsiY.T)
            self.L = clf.coef_.T

        return self.L

    def save(self, path: str) -> None:
        if self.L is None:
            print(f"'L' in '{self.__class__.__name__}' is no data.")
        else:
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(path, X=self.L)