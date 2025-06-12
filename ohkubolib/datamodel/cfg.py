import dataclasses as dc
from collections.abc import Callable
from typing import Any


class Singleton(object):
    def __new__(cls: type['Singleton'], *args: Any, **kargs: Any) -> Any:
        if not hasattr(cls, "_instance"):
            setattr(cls, "_instance", super(Singleton, cls).__new__(cls))
        return getattr(cls, "_instance")


@dc.dataclass(slots=True)
class DataModelConfig(Singleton):
    type_check: bool
    comment: Callable[[Any], Any] | None


@dc.dataclass(frozen=True, slots=True)
class GlobalConfig(Singleton):
    dataclass: dict[str, Any]
    pydantic: dict[str, Any]
    json: dict[str, Any]
    datamodel: DataModelConfig


# jsonに'__comment__'を追加
# デフォルトでは型情報を追加
def comment(obj: Any) -> dict[str, Any]:
    return {
        'type': obj.__class__.__name__,
    }


global_config = GlobalConfig(
    dataclass={
        'frozen': False,                  # 再代入を禁止
        'slots': True,                    # 後からselfに変数を生やすことを禁止
        'kw_only': False,                 # コンストラクタでキーワード引数のみを使う（a=1, b=2みたいな）
    },
    pydantic={
        'arbitrary_types_allowed': True,  # 非プリミティブ型を許可（np.ndarrayなど）
        'strict': True,                   # キャストしない（デフォルトでは'1'を1に変換する）
    },
    json={
        'indent': 4,
        'ensure_ascii': False,            # 日本語を使うのならば必須
    },
    datamodel=DataModelConfig(
        type_check=True,                  # 型チェックを有効にする (pydanticが必要)
        comment=comment,
    ),
)
