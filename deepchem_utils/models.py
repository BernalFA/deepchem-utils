from typing import Callable, Dict

from deepchem_utils.config import ModelsEnum


def make_constructor(cls, default_kwargs):
    def constructor(**kwargs):
        merged = {**default_kwargs, **kwargs}
        return cls(**merged)
    return constructor


class ModelFactory:
    def __init__(self):
        self._registry: Dict[str, Callable] = {}
        self._setup()

    def register(self, name: str, constructor: Callable):
        self._registry[name] = constructor

    def create(self, name: str, **kwargs):
        if name not in self._registry:
            raise ValueError(f"Model '{name}' not registered.")
        # add mode
        return self._registry[name](**kwargs)

    def _setup(self):
        for model_enum in ModelsEnum:
            self._registry[model_enum.name] = make_constructor(
                model_enum.cls, model_enum.defaults
            )
