from itertools import chain

import sympy

from .model_list import Models
from .sde import BaseBuilder, DiffsType, DriftsType, SDEModelBuilder


@SDEModelBuilder.register(Models.NoisyLotkaVolterra)
class LVBuilder(BaseBuilder):
    def check_dim(self) -> None:
        pass

    def set_params(self) -> list[tuple[sympy.Symbol, float]]:
        epsilons, mus, sigmas = self.params['epsilons'], self.params['mus'], self.params['sigmas']

        self.sym_epsilons = [sympy.Symbol(rf'\epsilon_{i+1}') for i in range(self.dim)]
        self.sym_sigmas = [sympy.Symbol(rf'\sigma_{i+1}') for i in range(self.dim)]
        self.sym_mus = [[sympy.Symbol(rf'\mu_{i+1},{j+1}') for j in range(self.dim)] for i in range(self.dim)]

        param_list: list[tuple[sympy.Symbol, float]] = []
        param_list.extend(zip(self.sym_epsilons, epsilons))
        param_list.extend(zip(chain.from_iterable(self.sym_mus), chain.from_iterable(mus)))
        param_list.extend(zip(self.sym_sigmas, sigmas))

        return param_list

    def set_terms(self) -> tuple[DriftsType, DiffsType]:
        drifts: DriftsType = [0]*self.dim
        diffs: DiffsType = [[0]*self.dim for _ in range(self.dim)]

        for d1 in range(self.dim):
            drifts[d1] = self.sym_epsilons[d1]*self.sym_xs[d1]
            for d2 in range(self.dim):
                drifts[d1] += self.sym_mus[d1][d2]*self.sym_xs[d2]*self.sym_xs[d1]
            diffs[d1][d1] = self.sym_sigmas[d1]*self.sym_xs[d1]

        return drifts, diffs
