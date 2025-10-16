from typing import Callable, Dict

from deepchem.feat import (
    ConvMolFeaturizer, DMPNNFeaturizer, DummyFeaturizer, GroverFeaturizer,
    MolGraphConvFeaturizer, SmilesToSeq,
)


_FEATURIZERS = [
    ConvMolFeaturizer,
    DMPNNFeaturizer,
    DummyFeaturizer,
    GroverFeaturizer,
    MolGraphConvFeaturizer,
    SmilesToSeq,
]


def make_constructor(cls):
    return lambda **kwargs: cls(**kwargs)


class FeaturizerFactory:
    def __init__(self):
        self._registry: Dict[str, Callable] = {}
        self._setup()

    def register(self, name: str, constructor: Callable):
        self._registry[name] = constructor

    def create(self, name: str, **kwargs):
        if name not in self._registry:
            raise ValueError(f"Featurizer '{name}' not registered.")
        return self._registry[name](**kwargs)

    def _setup(self):
        for featurizer in _FEATURIZERS:
            self._registry[featurizer.__name__] = make_constructor(featurizer)
