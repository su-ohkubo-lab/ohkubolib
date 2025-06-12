from typing import Any

import numpy as np
import sympy

from ohkubolib.datamodel import datamodel, field
from ohkubolib.koopman.dictionaries import KronMonomial, StateDictionary
from ohkubolib.model import SDEModel
from ohkubolib.time_evolution import TEConfig


def get_adjoint(model: SDEModel) -> sympy.Expr:
    """
    演算子Lの随伴L^daggerをsympyで計算する。
    近似ではないので、辞書には依存しない。
    """
    Mat_drifts = sympy.Matrix(model.drifts)
    Mat_diffs = sympy.Matrix(model.diffs)
    B = sympy.Rational(1,2)*Mat_diffs * Mat_diffs.transpose()

    L = 0
    for d1 in range(model.dim):
        L = L + Mat_drifts[d1]*model.sym_derivs[d1]
        for d2 in range(model.dim):
            if B[d1,d2] != 0:
                L = L + model.sym_derivs[d1]*model.sym_derivs[d2]*B[d1,d2]

    L = sympy.expand(L)
    return L


def create_operator(model: SDEModel, n_trunc: int) -> np.ndarray:
    """
    SDEから生成子行列を作成する。
    kronecker積ベースのmonomialを想定した微分を行う。
    """
    L = get_adjoint(model)

    # 方程式の数
    n_eqs = np.power(n_trunc, model.dim)
    # 空の行列を作る
    A = np.zeros((n_eqs,n_eqs), dtype='float64')

    ### 変数と偏微分の記号の設定。これらの次数のリストを得る。
    variables = []
    variables.extend(model.sym_xs)
    variables.extend(model.sym_derivs)

    dim = model.dim

    ### 式に具体的なパラメータを代入し、オペレータを求める。
    expand = L.subs(model.params_tuple)
    for e, arg in enumerate(expand.args):   # 各項（イベント）ごとに処理
        coeff = sympy.LC(arg) # 係数を引っ張り出す
        degree_list = np.array(sympy.degree_list(arg, gens=variables)) # 次数のリストを得る
        state_change = degree_list[:dim] - degree_list[dim:] # 状態変化のベクトルを得る
        state_rate = degree_list[dim:] # 微分演算子の部分の次数を得る（これが係数に反映される）
        # 微分演算子の回数に応じて処理を変える（定数項、ドリフト項、拡散項）
        op = 0.0
        for d in range(dim):
            if state_rate[d] == 0: # 定数項
                comp = np.eye(n_trunc,k=-state_change[d])
            elif state_rate[d] == 1: # ドリフト項
                comp = np.eye(n_trunc,k=-state_change[d])@np.diag(np.arange(n_trunc))
            else: # 拡散項
                comp = np.eye(n_trunc,k=-state_change[d])@np.diag(np.arange(n_trunc)*(np.arange(n_trunc)-1))
            if d == 0: # 最初は自分自身のみ
                op = coeff*comp
            else: # 2次元目以降はクロネッカー積
                op = np.kron(comp, op)
        A = A + op
    # あとの数値計算のために dtype を変更
    A = np.array(A, dtype='float64')
    return A


@datamodel(frozen=False)
class Dual:
    dim: int
    order: int
    comp_index: list[int]

    psi: StateDictionary = field(init=False, default=None)

    stats: np.ndarray = field(init=False, default=None)
    p_list: list[np.ndarray] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.psi = KronMonomial(dim=self.dim, order=self.order)

    def __call__(self, model: SDEModel, config: TEConfig) -> np.ndarray:
        if self.dim != model.dim:
            raise ValueError("Dimension mismatch: {0} != {1}".format(self.dim, model.dim))

        calc_stat = lambda p: np.dot(p, self.psi(config.x0))

        A = create_operator(model, self.order+1)
        n_eqs = A.shape[0]

        p_ini = np.zeros(n_eqs)
        p_ini[self.psi.s2i(np.array(self.comp_index))] = 1.0

        stats = [calc_stat(p_ini)]
        p_list = [p_ini]
        sol = p_ini
        time_list = config.time_list
        dt = config.dt
        for i, time in enumerate(time_list.dt[1:]):
            leftA = np.eye(n_eqs) - 0.5*dt*A
            right_b = (np.eye(n_eqs) + 0.5*dt*A)@sol
            sol = np.linalg.solve(leftA ,right_b)

            if time_list.is_obs_indices(i+1):
                stats.append(calc_stat(sol))
                p_list.append(sol.copy())

        self.stats = np.array(stats)
        self.p_list = p_list
        return self.stats


@datamodel(frozen=False)
class DualKoopman:
    dim: int
    order: int

    psi: StateDictionary = field(init=False, default=None)

    K: np.ndarray = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.psi = KronMonomial(dim=self.dim, order=self.order)

    def __call__(self, model: SDEModel, config: TEConfig) -> np.ndarray:
        if self.dim != model.dim:
            raise ValueError("Dimension mismatch: {0} != {1}".format(self.dim, model.dim))

        m = self.psi.num_states
        K = np.zeros((m, m), dtype='float64')
        solver = Dual(dim=self.dim, order=self.order, comp_index=[0]*self.dim)
        for i in range(m):
            stat = self.psi.i2s(i).tolist()
            solver.comp_index = stat
            _ = solver(model, config)
            K[self.psi.s2i(stat), :] = solver.p_list[1]

        self.K = K
        return self.K
