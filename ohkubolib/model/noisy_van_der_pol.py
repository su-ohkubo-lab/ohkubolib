import sympy

from .model_list import Models
from .sde import BaseBuilder, DiffsType, DriftsType, SDEModelBuilder


@SDEModelBuilder.register(Models.NoisyVanDerPol)
class VdPBuilder(BaseBuilder):
    def check_dim(self) -> None:
        if self.dim != 2:
            raise ValueError

    def set_params(self) -> list[tuple[sympy.Symbol, float]]:
        epsilon, nu11, nu22 = self.params['epsilon'], self.params['nu11'], self.params['nu22']

        self.sym_epsilon = sympy.Symbol(r'\epsilon')
        self.sym_nu11 = sympy.Symbol(r'\nu_{11}')
        self.sym_nu22 = sympy.Symbol(r'\nu_{22}')

        param_list: list[tuple[sympy.Symbol, float]] = [
            (self.sym_epsilon, epsilon),
            (self.sym_nu11, nu11),
            (self.sym_nu22, nu22)
        ]

        return param_list

    def set_terms(self) -> tuple[DriftsType, DiffsType]:
        drifts = [
            self.sym_xs[1],
            self.sym_epsilon*(1.0 - self.sym_xs[0]**2)*self.sym_xs[1] - self.sym_xs[0]
        ]
        diffs = [[self.sym_nu11, 0.0], [0.0, self.sym_nu22]]

        return drifts, diffs
