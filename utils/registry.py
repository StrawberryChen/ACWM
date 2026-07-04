from collections.abc import Callable
from typing import Any


class Registry:
    def __init__(self):
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, constructor: Callable[..., Any]) -> None:
        if name in self._items:
            raise KeyError(f"duplicate registry entry: {name}")
        self._items[name] = constructor

    def build(self, specification: dict[str, Any]) -> Any:
        specification = dict(specification)
        name = specification.pop("name")
        if name not in self._items:
            raise KeyError(f"unknown component {name!r}; available: {sorted(self._items)}")
        return self._items[name](**specification)

