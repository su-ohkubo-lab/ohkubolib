import dataclasses as dc
from typing import Any

import numpy as np


def force_set_attr(obj: Any, attr: str, value: Any) -> None:
    """
    Forcefully set an attribute of an object, even if it's read-only.
    """
    object.__setattr__(obj, attr, value)


def pprint(obj: Any, indent: int = 2, depth: int | None = None, jupyter_integrate: bool = True) -> None:
    if not dc.is_dataclass(obj):
        raise TypeError(f'{obj} must be a dataclass')

    opt = np.get_printoptions()
    np.set_printoptions(threshold=100)

    if jupyter_integrate:
        from IPython.display import display_pretty
        display_pretty(_pprint_inner(obj, indent, depth), raw=True)
    else:
        print(_pprint_inner(obj, indent, depth))

    np.set_printoptions(**opt)


def _pprint_inner(obj: Any, indent: int, depth: int | None, current: int = 0) -> str:
    indent_str = ' ' * indent * current
    field_indent_str = ' ' * indent * (current + 1)

    if dc.is_dataclass(obj):
        if depth is not None and current >= depth:
            return f'{type(obj).__name__}(...)'

        strlist = [type(obj).__name__ + '(']
        for field in dc.fields(obj):
            child = _pprint_inner(getattr(obj, field.name), indent, depth, current + 1)
            strlist.append(field_indent_str + field.name + '=' + child + ',')
        strlist.append(indent_str + ')')
        return '\n'.join(strlist)

    elif isinstance(obj, list):
        if depth is not None and current >= depth:
            return '[...]'

        prefix, suffix = '[', ']'
        strlist = []
        for item in obj:
            child = _pprint_inner(item, indent, depth, current + 1)
            strlist.append(child)

        oneline = prefix + ', '.join(strlist) + suffix
        if len(oneline) <= 80:
            return oneline
        else:
            suffix = indent_str + suffix
            for i in range(len(strlist)):
                strlist[i] = field_indent_str + strlist[i]
            return prefix + '\n' + ',\n'.join(strlist) + '\n' + suffix

    elif isinstance(obj, dict):
        if depth is not None and current >= depth:
            return '{...}'

        prefix, suffix = '{', '}'
        strlist = []
        for key, value in obj.items():
            key_str = _pprint_inner(key, indent, depth, current + 1)
            value_str = _pprint_inner(value, indent, depth, current + 1)
            strlist.append(key_str + ': ' + value_str)

        oneline = prefix + ', '.join(strlist) + suffix
        if len(oneline) <= 80:
            return oneline
        else:
            suffix = indent_str + suffix
            for i in range(len(strlist)):
                strlist[i] = field_indent_str + strlist[i] + ','
            return prefix + '\n' + '\n'.join(strlist) + '\n' + suffix

    else:
        elem_str = str(obj)
        lines = elem_str.split('\n')
        if len(lines) > 1:
            for i in range(1, len(lines)):
                lines[i] = field_indent_str + lines[i]
            return '\n'.join(lines)
        else:
            return elem_str


