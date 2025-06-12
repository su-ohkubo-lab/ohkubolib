import abc
import copy
import dataclasses as dc
import importlib
import json
import sys
from collections.abc import Callable
from functools import cached_property
from pathlib import Path
from typing import Any, Generic, Self, TypeVar, dataclass_transform

import typing_extensions

from .cfg import global_config
from .typing import expand_type

_MODEL_CONFIG = '__datamodel_config__'
_FIELD_CONFIG = 'datamodel'


@dc.dataclass
class SerializeContext:
    obj_path: list[str | int]
    save_dir: Path
    asset_dir: str
    dc_typed: bool

    def next(self, obj: Any, relative_path_to_obj: list[str | int] = []) -> Any:
        return _asdict_field(obj, dc.replace(self, obj_path=[*self.obj_path, *relative_path_to_obj]))

    def _add_path(self, path: str | int) -> Self:
        return dc.replace(self, obj_path=[*self.obj_path, path])


_D = TypeVar('_D')
@dc.dataclass
class DeserializeContext:
    target_type: type
    save_dir: Path
    dc_typed: bool

    def next(self, type_: type[_D], obj: Any) -> _D:
        return _fromdict_field(obj, dc.replace(self, target_type=type_))


class SerdeBase(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def serialize(obj: Any, ctx: SerializeContext) -> Any:
        pass

    @staticmethod
    @abc.abstractmethod
    def deserialize(obj: Any, ctx: DeserializeContext) -> Any:
        pass


class DefaultSerde:
    __class: dict[type, type[SerdeBase]] = {}
    __subclasses: dict[type, type[SerdeBase]] = {}

    @classmethod
    def register(cls, type_: type, include_subclass: bool = False) -> Callable[[Any], Any]:
        def wrapper(cls_: type) -> Any:
            if not issubclass(cls_, SerdeBase):
                raise TypeError(f'{cls_} must be a subclass of SerdeBase')
            if include_subclass:
                cls.__subclasses[type_] = cls_
            else:
                cls.__class[type_] = cls_
            return cls_

        return wrapper

    @classmethod
    def get(cls, type_: type) -> type[SerdeBase] | None:
        if type_ in cls.__class:
            return cls.__class[type_]
        else:
            # some types will raise TypeError in issubclass
            # e.g. typing.Literal
            try:
                issubclass(type_, type)
            except TypeError:
                return None

            for base, serde in cls.__subclasses.items():
                if issubclass(type_, base):
                    return serde

        return None



_S = TypeVar('_S', bound=SerdeBase)

@dc.dataclass
class DataModelConfig(Generic[_S]):
    serde: type[_S] | None

@dc.dataclass
class DataModelFieldConfig(Generic[_S]):
    serde: type[_S] | None
    ignore_serde: bool


def _get_model_config(cls: type) -> DataModelConfig | None:
    if (conf := getattr(cls, _MODEL_CONFIG, None)) is not None:
        if isinstance(conf, DataModelConfig):
            return conf

    return None

def _get_field_config(field: dc.Field) -> DataModelFieldConfig | None:
    if (conf := field.metadata.get(_FIELD_CONFIG)) is not None:
        if isinstance(conf, DataModelFieldConfig):
            return conf

    return None


def _asdict(obj: Any, ctx: SerializeContext) -> Any:
    assert (not isinstance(obj, type)) and dc.is_dataclass(obj)

    obj_type = type(obj)
    model_conf = _get_model_config(obj_type)

    if model_conf is not None and model_conf.serde is not None:
        return model_conf.serde.serialize(obj, ctx)
    else:
        dic = {}
        for field in dc.fields(obj):
            field_ser = _asdict_field
            type_ = field.type

            if dc.is_dataclass(type_):
                assert isinstance(type_, type)
                field_ser = _asdict

            elif isinstance(type_, str):
                field_ser = _asdict_field

            elif (default := DefaultSerde.get(type_)) is not None:
                field_ser = default.serialize

            config = _get_field_config(field)
            if config is not None:
                if config.serde is not None:
                    field_ser = config.serde.serialize

                if config.ignore_serde:
                    continue

            dic[field.name] = field_ser(getattr(obj, field.name), ctx._add_path(field.name))

        if ctx.dc_typed:
            wrapper = {
                '__type__': obj_type.__name__,
                '__fields__': dic
            }

            return wrapper
        else:
            return dic


def _asdict_field(obj: Any, ctx: SerializeContext) -> Any:
    obj_type = type(obj)

    if dc.is_dataclass(obj_type):
        return _asdict(obj, ctx)

    elif issubclass(obj_type, (list, tuple)):
        return [
            _asdict_field(v, ctx._add_path(i))
            for i, v in enumerate(obj)
        ]

    elif issubclass(obj_type, dict):
        return {
            _asdict_field(k, ctx): _asdict_field(v, ctx._add_path(k))
            for k, v in obj.items()
        }

    elif (default := DefaultSerde.get(obj_type)) is not None:
        return default.serialize(obj, ctx)
    else:
        return copy.deepcopy(obj)


def _fromdict(obj: dict[str, Any], ctx: DeserializeContext) -> Any:
    cls = ctx.target_type
    assert dc.is_dataclass(cls)


    model_conf = _get_model_config(cls)

    if model_conf is not None and model_conf.serde is not None:
        return model_conf.serde.deserialize(obj, ctx)
    else:
        dic = {}
        fields: dict[str, dc.Field] = {field.name: field for field in dc.fields(cls)}

        if ctx.dc_typed:
            if '__fields__' not in obj:
                raise ValueError('obj must have __fields__ key when dataclass typed is True')

            obj = obj['__fields__']

        for k, v in obj.items():
            field = fields.get(k)

            if field is None:
                dic[k] = v
            else:
                field_des = _fromdict_field
                type_ = field.type

                if (default := DefaultSerde.get(type_)) is not None:
                    field_des = default.deserialize

                config = _get_field_config(field)
                if config is not None:
                    if config.serde is not None:
                        field_des = config.serde.deserialize
                    if config.ignore_serde:
                        continue

                ctx = dc.replace(ctx, target_type=type_)
                dic[k] = field_des(v, ctx)


        return cls(**dic)


def _fromdict_field(obj: Any, ctx: DeserializeContext) -> Any:
    contains_dc = False
    error = None

    for obj_type, args in expand_type(ctx.target_type):
        if dc.is_dataclass(obj_type):
            contains_dc = True
            try:
                return _fromdict(obj, dc.replace(ctx, target_type=obj_type))
            except Exception as e:
                error = e
                continue

        elif issubclass(obj_type, tuple):
            if not isinstance(obj, list):
                continue

            res = []
            if args is None:
                res = obj
            else:
                for i, arg in enumerate(args):
                    ctx = dc.replace(ctx, target_type=arg)
                    if i >= len(obj):
                        break
                    res.append(_fromdict_field(obj[i], ctx))

            return obj_type(res)

        elif issubclass(obj_type, list):
            if not isinstance(obj, list):
                continue

            ctx = dc.replace(ctx, target_type=type if args is None else args[0])
            return obj_type(
                _fromdict_field(v, ctx)
                for v in obj
            )

        elif issubclass(obj_type, dict):
            if not isinstance(obj, dict):
                continue

            key_type = val_type = type(object)

            if args is not None:
                if len(args) > 1:
                    key_type, val_type = args[0], args[1]
                else:
                    key_type = args[0]

            return obj_type(
                (_fromdict_field(k, dc.replace(ctx, target_type=key_type)), _fromdict_field(v, dc.replace(ctx, target_type=val_type)))
                for k, v in obj.items()
            )

        elif (default := DefaultSerde.get(obj_type)) is not None:
            return default.deserialize(obj, ctx)
        else:
            continue

    # dataclassが含まれていて，どれにもマッチしなかった場合は例外を投げる
    if contains_dc and error is not None:
        raise error

    return obj


_COMMENT_KEY = '__comment__'

def save_json(obj: Any, path: str, typed: bool = True) -> None:
    if not dc.is_dataclass(obj):
        raise ValueError('obj must be a dataclass')

    json_path = Path(path)
    asset_dir = json_path.stem

    dic = _asdict(obj, SerializeContext(obj_path=[], save_dir=json_path.parent, asset_dir=asset_dir, dc_typed=typed))

#    if global_config.datamodel.comment is not None:
#        dic = { _COMMENT_KEY: global_config.datamodel.comment(obj) } | dic

    json_path.write_text(json.dumps(dic, **global_config.json), encoding='utf-8')


_U = TypeVar('_U')
def load_json(cls: type[_U], path: str, typed: bool = True) -> _U:
    if not dc.is_dataclass(cls):
        raise ValueError('cls must be a dataclass')

    json_path = Path(path)
    dic = json.loads(json_path.read_text(encoding='utf-8'))

    if _COMMENT_KEY in dic:
        del dic[_COMMENT_KEY]

    return _fromdict(dic, DeserializeContext(target_type=cls, save_dir=json_path.parent, dc_typed=typed))


def field(serde: type[_S] | None = None, ignore_serde: bool = False, **kwargs): # type: ignore
    if 'metadata' not in kwargs:
        kwargs['metadata'] = {}

    kwargs['metadata'][_FIELD_CONFIG] = DataModelFieldConfig(serde=serde, ignore_serde=ignore_serde)

    return dc.field(**kwargs)


def _dyn_import(name: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    elif (spec := importlib.util.find_spec(name)) is not None: # type: ignore[attr-defined]
        # If you chose to perform the actual import ...
        module = importlib.util.module_from_spec(spec) # type: ignore[attr-defined]
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    else:
        raise ModuleNotFoundError(name=name)


def _get_dataclass(type_validate: bool, **kwargs: Any) -> Any:
    dc_conf = global_config.dataclass
    if len(kwargs) > 0:
        dc_conf = dc_conf | kwargs

    if type_validate:
        pd = _dyn_import('pydantic')
        return pd.dataclasses.dataclass(config=pd.ConfigDict(**global_config.pydantic), **dc_conf)
    else:
        return dc.dataclass(**dc_conf)



# slotsが有効でcached_propertyがある場合は特殊な処理が必要
def _preprocess_class(cls: type, slots: bool) -> dict[str, Any]:
    if not slots:
        return {}
    else:
        if not hasattr(cls, '__dict__'):
            raise ValueError(f'{cls} not have __dict__. Maybe using other decorator?')

        # cached_propertyを取り出しておいて実体を削除
        # slotsにプロパティ名を追加するために__annotations__に追加
        cp = {
            name: value.func
            for name, value in cls.__dict__.items()
            if isinstance(value, cached_property)
        }

        for name in cp.keys():
            delattr(cls, name)
            cls.__annotations__[name] = Any
            setattr(cls, name, dc.field(init=False, repr=False, compare=False))

        return cp


def _process_class(cls: type, serde: type[_S] | None, cp: dict[str, Any], type_validate: bool) -> None:
    setattr(cls, _MODEL_CONFIG, DataModelConfig(serde=serde))

    # slotsが有効でcached_propertyがある場合は特殊な処理が必要
    if hasattr(cls, '__slots__') and len(cp) > 0:
        # slotsにプロパティ名を追加するために設定したダミーを削除
        # __dataclass_fields__にフィールドとして追加されてしまうので削除
        for name in cp.keys():
            del cls.__annotations__[name]
            del cls.__dataclass_fields__[name] # type: ignore

        if type_validate:
            # pydanticを使う場合には再構築が必要
            res = _dyn_import('pydantic').dataclasses.rebuild_dataclass(cls, force=True)
            if res is None or not res:
                raise ValueError('Failed to rebuild dataclass')

        # __getattr__で関数を呼び出して値を取得して返す
        # その際setを呼び出して値をキャッシュする (frozen=Trueでも使える抜道を使っている)
        # この時slotsにプロパティ名がないとAttributeErrorが発生するのでここまでの処理が必要
        def getattr(self, name: str, cached_prop=cp) -> Any: # type: ignore
            func = cached_prop.get(name)
            if func is not None:
                result = func(self)
                setter = object.__setattr__.__get__(self)
                setter(name, result)
                return result
            else:
                return object.__getattribute__(self, name)

        cls.__getattr__ = getattr # type: ignore


_T = TypeVar('_T')

@dataclass_transform(field_specifiers=(field, dc.field, dc.Field))
def datamodel(_cls: type[_T] | None = None, *, serde: type[_S] | None = None, **kwargs): # type: ignore
    def wrap(cls: type[_T]) -> type[_T]:
        type_validate = global_config.datamodel.type_check

        slots = kwargs.get('slots', global_config.dataclass.get('slots', False))
        cp = _preprocess_class(cls, slots)
        cls = _get_dataclass(type_validate, **kwargs)(cls)
        _process_class(cls, serde, cp, type_validate)
        return cls

    if _cls is None:
        return wrap
    else:
        return wrap(_cls)
