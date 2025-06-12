from enum import Enum
from typing import Any

import numpy as np

from .core import (
    _COMMENT_KEY,
    DefaultSerde,
    DeserializeContext,
    SerdeBase,
    SerializeContext,
)


def _asset_file_fullpath(ctx: SerializeContext, ext: str) -> str:
    fullpath = ctx.save_dir / _asset_file_path(ctx, ext)
    fullpath.parent.mkdir(parents=True, exist_ok=True)
    return str(fullpath)


def _asset_file_path(ctx: SerializeContext, ext: str) -> str:
    return ctx.asset_dir + '/' + '_'.join([str(p) for p in ctx.obj_path]) + ext


def _asset_dict(obj: Any, ctx: SerializeContext, ext: str) -> dict[str, Any]:
    return {
        _COMMENT_KEY: {
            'type': obj.__class__.__name__,
        },
        'path': _asset_file_path(ctx, ext)
    }


def _resolve_dict(obj: Any, ctx: DeserializeContext) -> str:
    if not isinstance(obj, dict):
        raise ValueError('Invalid asset object')

    if 'path' not in obj:
        raise ValueError('Invalid asset dictionary')

    return str(ctx.save_dir / obj['path'])


@DefaultSerde.register(Enum, include_subclass=True)
class EnumSerde(SerdeBase):
    @staticmethod
    def serialize(v: Enum, ctx: SerializeContext) -> str:
        return v.name

    @staticmethod
    def deserialize(v: str, ctx: DeserializeContext) -> Enum:
        return ctx.target_type[v] # type: ignore


@DefaultSerde.register(np.ndarray)
class NpzSerde(SerdeBase):
    @staticmethod
    def serialize(v: np.ndarray, ctx: SerializeContext) -> dict[str, Any]:
        file_path = _asset_file_fullpath(ctx, '.npz')
        np.savez_compressed(file_path, v)

        dic = _asset_dict(v, ctx, '.npz')
        dic[_COMMENT_KEY]['shape'] = v.shape
        dic[_COMMENT_KEY]['dtype'] = str(v.dtype)
        return dic

    @staticmethod
    def deserialize(v: Any, ctx: DeserializeContext) -> np.ndarray:
        path = _resolve_dict(v, ctx)
        return np.load(path)['arr_0']


class NdArryaListSerde(SerdeBase):
    @staticmethod
    def serialize(v: np.ndarray, ctx: SerializeContext) -> list:
        return v.tolist()

    @staticmethod
    def deserialize(v: Any, ctx: DeserializeContext) -> np.ndarray:
        return np.array(v)
