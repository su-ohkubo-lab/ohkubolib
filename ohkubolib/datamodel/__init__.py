from dataclasses import replace

import ohkubolib.datamodel.serde

from .cfg import global_config
from .core import (
    DefaultSerde,
    DeserializeContext,
    SerdeBase,
    SerializeContext,
    datamodel,
    field,
    load_json,
    save_json,
)
from .utils import force_set_attr, pprint
