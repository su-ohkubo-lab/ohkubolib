from __future__ import annotations
from typing import Literal
import inspect
import pathlib
import numpy as np
import scipy.linalg as LA
from ohkubolib.koopman.dictionaries import Dictionary
from ohkubolib.koopman.edmd import EDMD

# Koopman Lifting のクラス (EDMD のクラスを継承)
class Lifting(EDMD):
    dictionary_X: Dictionary # x ベクトルに対する辞書
    dictionary_Y: Dictionary # y ベクトルに対する辞書 (Lifting クラスの場合, x ベクトルに対する辞書と同じ)
    dt: int | float          # 時系列データの時間間隔
    K: np.ndarray | None     # データから求められる Koopman 行列 (初期値: None)
    L: np.ndarray | None     # データから求められる Koopman 生成子行列 (初期値: None)
    def __init__(self, dictionary: Dictionary, dt: int | float = 1.0e-3) -> None:
        super(Lifting, self).__init__(dictionary=dictionary)

        if type(dt) is not int and type(dt) is not float:
            raise ValueError(f"'dt' in '{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}()' is expected type '<class 'int'>' or '<class 'float'>', but type '{type(dt)}'.")

        self.dt = dt
        self.K = None
        self.L = None

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        self.K = super(Lifting, self).__call__(X, Y)
        self.L = LA.logm(self.K) / self.dt
        return self.L

    def save(self, path: str, data: Literal['L', 'K'] = 'L') -> None:
        if data == 'L':
            if self.L is None:
                print(f"'L' in '{self.__class__.__name__}' is no data.")
            else:
                pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(path, X=self.L)

        elif data == 'K':
            if self.K is None:
                print(f"'K' in '{self.__class__.__name__}' is no data.")
            else:
                pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(path, X=self.K)

        else:
            raise ValueError(f"'data' in '{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}()' is expected 'L' or 'K', but '{data}'.")