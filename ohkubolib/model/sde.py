import abc
from collections.abc import Callable
from functools import cached_property
from typing import Any

import sympy

import ohkubolib.datamodel as dm

from .model_list import Models


@dm.datamodel
class SDEModelInfo:
    """
    Model information to create SDEModel.
    """
    type: Models
    name: str
    dim: int
    params: dict[str, Any]
    is_ode: bool


type DriftsType = list[int | float | sympy.Expr | sympy.Symbol]
type DiffsType = list[DriftsType]

class BaseBuilder(abc.ABC):
    """
    Base class for SDE model builders.
    Don't use this class directly.
    """
    def __init__(self, info: SDEModelInfo) -> None:
        self.info = info
        self.dim = info.dim
        self.params = info.params

        self.sym_xs = [sympy.symbols(f'x{i+1}') for i in range(self.dim)]
        self.sym_dws = [sympy.symbols(f'dW{i+1}') for i in range(self.dim)]
        self.sym_derivs = [sympy.symbols(f'dx{i+1}') for i in range(self.dim)]

    @abc.abstractmethod
    def check_dim(self) -> None:
        pass

    @abc.abstractmethod
    def set_params(self) -> list[tuple[sympy.Symbol, float]]:
        pass

    @abc.abstractmethod
    def set_terms(self) -> tuple[DriftsType, DiffsType]:
        pass

    def build(self) -> 'SDEModel':
        self.check_dim()
        param_list = self.set_params()
        drifts, diffs = self.set_terms()

        return SDEModel(
            info=self.info,
            sym_xs=self.sym_xs,
            sym_dws=self.sym_dws,
            sym_derivs=self.sym_derivs,
            drifts=drifts,
            diffs=diffs,
            params_tuple=param_list,
        )


class SDEModelBuilder:
    """
    Factory class for SDEModel.
    This class is used to create SDEModel instances from SDEModelInfo.
    """
    class_: dict[Models, type[BaseBuilder]] = {}

    @classmethod
    def register(cls, model_name: Models) -> Callable:
        def wrapper(cls_: type[BaseBuilder]) -> type[BaseBuilder]:
            if not issubclass(cls_, BaseBuilder):
                raise TypeError(f"Model Builder {model_name} must be a subclass of BaeeBuilder")
            cls.class_[model_name] = cls_
            return cls_

        return wrapper

    @classmethod
    def create(cls, type: Models, name: str, dim: int, params: dict[str, Any], is_ode: bool) -> 'SDEModel':
        return cls.create_from(info=SDEModelInfo(
            type=type,
            name=name,
            dim=dim,
            params=params,
            is_ode=is_ode
        ))

    @classmethod
    def create_from(cls, info: SDEModelInfo) -> 'SDEModel':
        builder = cls.class_[info.type](info=info)
        return builder.build()


class SerdeSDEModel(dm.SerdeBase):
    """
    Definition of serialization and deserialization for SDEModel.
    """
    @staticmethod
    def serialize(obj: 'SDEModel', ctx: dm.SerializeContext) -> dict[str, Any]:
        return ctx.next(obj.info)

    @staticmethod
    def deserialize(obj: Any, ctx: dm.DeserializeContext) -> 'SDEModel':
        model_info: SDEModelInfo = ctx.next(SDEModelInfo, obj)
        return SDEModelBuilder.create_from(model_info)


@dm.datamodel(serde=SerdeSDEModel)
class SDEModel:
    info: SDEModelInfo

    sym_xs: list[sympy.Symbol]
    sym_dws: list[sympy.Symbol]
    sym_derivs: list[sympy.Symbol]

    drifts: DriftsType
    diffs: DiffsType

    params_tuple: list[tuple[sympy.Symbol, float]]

    @property
    def dim(self) -> int:
        return self.info.dim

    def with_name(self, name: str) -> 'SDEModel':
        info = dm.replace(self.info, name=name)
        return dm.replace(self, info=info)

    @cached_property
    def drifts_func(self) -> Callable:
        args = ([self.sym_xs])
        sym_func = sympy.Matrix(self.drifts).subs(self.params_tuple)
        return sympy.lambdify(args, sym_func, "numpy")


    @cached_property
    def diffs_func(self) -> Callable:
        args = ([self.sym_xs])
        sym_func = sympy.Matrix(self.diffs).subs(self.params_tuple)
        return sympy.lambdify(args, sym_func, "numpy")


    def print_model(self) -> None:
        from IPython.display import Math, display

        print(self.info.name, '(ODE)' if self.info.is_ode else '(SDE)')

        sympy.init_printing()

        for d1 in range(self.dim):
            latex_txt = r'd{0}(t) = \left({1}\right) dt'.format(sympy.latex(self.sym_xs[d1]), sympy.latex(self.drifts[d1]))
            if not self.info.is_ode:
                for d2 in range(self.dim):
                    if self.diffs[d1][d2] != 0:
                        latex_txt += r'+\left({0}\right) {1}'.format(sympy.latex(self.diffs[d1][d2]), sympy.latex(self.sym_dws[d2]))
            display(Math(latex_txt))


    def print_subs_model(self) -> None:
        from IPython.display import Math, display

        print(self.info.type, '(ODE)' if self.info.is_ode else '(SDE)')
        print('name:', self.info.name)

        sympy.init_printing()

        for d1 in range(self.dim):
            drift = self.drifts[d1]
            if isinstance(drift, sympy.Expr):
                drift = drift.subs(self.params_tuple)

            latex_txt = r'd{0}(t) = \left({1}\right) dt'.format(sympy.latex(self.sym_xs[d1]), sympy.latex(drift))
            if not self.info.is_ode:
                for d2 in range(self.dim):
                    diff = self.diffs[d1][d2]
                    if isinstance(diff, sympy.Expr):
                        diff = diff.subs(self.params_tuple)

                    latex_txt += r'+\left({0}\right) {1}'.format(sympy.latex(diff), sympy.latex(self.sym_dws[d2]))
            display(Math(latex_txt))
