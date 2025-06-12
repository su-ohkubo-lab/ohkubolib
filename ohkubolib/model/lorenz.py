import sympy

from .model_list import Models
from .sde import BaseBuilder, DiffsType, DriftsType, SDEModelBuilder


@SDEModelBuilder.register(Models.Lorenz)
class LZBuilder(BaseBuilder):
    def check_dim(self) -> None:
        if self.dim != 3:
            raise ValueError

    def set_params(self) -> list[tuple[sympy.Symbol, float]]:
        sigma, ro, beta = self.params['sigma'], self.params['ro'], self.params['beta']

        self.sym_sigma = sympy.Symbol(r'\sigma')
        self.sym_ro = sympy.Symbol(r'\ro')
        self.sym_beta = sympy.Symbol(r'\beta')

        param_list: list[tuple[sympy.Symbol, float]] = [
            (self.sym_sigma, sigma),
            (self.sym_ro, ro),
            (self.sym_beta, beta)
        ]

        return param_list

    def set_terms(self) -> tuple[DriftsType, DiffsType]:
        drifts = [
            self.sym_sigma * (self.sym_xs[1] - self.sym_xs[0]),
            self.sym_xs[0] * (self.sym_ro - self.sym_xs[2]) - self.sym_xs[1],
            self.sym_xs[0] * self.sym_xs[1] - self.sym_beta * self.sym_xs[2]
        ]
        diffs = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

        return drifts, diffs
