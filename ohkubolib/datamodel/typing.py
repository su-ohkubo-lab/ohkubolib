import types


def _is_generic_alias(type_: type) -> bool:
    return type(type_) is types.GenericAlias

def _is_union(type_: type) -> bool:
    return type(type_) is types.UnionType

def _inspect_ga(type_: type) -> tuple[type, tuple[type]] | None:
    origin = getattr(type_, '__origin__', None)
    args = getattr(type_, '__args__', None)

    # GenericAlias (e.g. list[int]) must have both origin (list) and args (int)
    if origin is None or args is None:
        return None
    else:
        return (origin, args)

def _expand_union(type_: type) -> tuple[type]:
    if _is_union(type_):
        args = getattr(type_, '__args__', None)
        if args is None:
            return (type_, )
        else:
            return args
    else:
        return (type_, )

# list[int] | str -> [(list, [int]), (str, None)]
def expand_type(type_: type) -> list[tuple[type, tuple[type] | None]]:
    result: list[tuple[type, tuple[type] | None]] = []
    for t in _expand_union(type_):
        if _is_generic_alias(t):
            info = _inspect_ga(t)
            if info is not None:
                origin, args = info
                result.append((origin, args))
            else:
                result.append((t, None))
        else:
            result.append((t, None))

    return result
