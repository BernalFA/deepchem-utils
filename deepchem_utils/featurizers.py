from typing import Callable, Dict

from deepchem.feat import (
    ConvMolFeaturizer, DMPNNFeaturizer, DummyFeaturizer, GroverFeaturizer,
    MolGraphConvFeaturizer, SmilesToSeq,
)


_FEATURIZERS = {
    "ConvMolFeaturizer": ConvMolFeaturizer,
    "DMPNNFeaturizer": DMPNNFeaturizer,
    "DummyFeaturizer": DummyFeaturizer,
    "GroverFeaturizer": GroverFeaturizer,
    "MolGraphConvFeaturizer": MolGraphConvFeaturizer,
    "SmilesToSeq": SmilesToSeq,
}


class FeaturizerFactory:
    def __init__(self):
        self._registry: Dict[str, Callable] = {}

    def register(self, name: str, constructor: Callable):
        self._registry[name] = constructor

    def create(self, name: str, **kwargs):
        if name not in self._registry:
            raise ValueError(f"Featurizer '{name}' not registered.")
        return self._registry[name](**kwargs)


# Register all available featurizers
featurizer_factory = FeaturizerFactory()
for featurizer_name, featurizer in _FEATURIZERS.items():
    featurizer_factory.register(
        featurizer_name,
        lambda featurizer=featurizer, **kwargs: featurizer(**kwargs)
    )
