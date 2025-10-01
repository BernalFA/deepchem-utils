from deepchem.feat import (
    MolGraphConvFeaturizer, ConvMolFeaturizer, DMPNNFeaturizer, DummyFeaturizer
)
from deepchem.models import GraphConvModel, DAGModel
from deepchem.models.torch_models import (
    MPNNModel, DMPNNModel, GCNModel, AttentiveFPModel, GATModel, Chemberta
)


FEATURIZERS = {
    "MolGraphConvFeaturizer": MolGraphConvFeaturizer(use_edges=True),
    "ConvMolFeaturizer": ConvMolFeaturizer(),
    "DMPNNFeaturizer": DMPNNFeaturizer(),
    "DummyFeaturizer": DummyFeaturizer(),
}


MODELS = {
    "GraphConvModel": GraphConvModel,
    "MPNNModel": MPNNModel,
    "DMPNNModel": DMPNNModel,
    "GCNModel": GCNModel,
    "AttentiveFPModel": AttentiveFPModel,
    "GATModel": GATModel,
    "DAGModel": DAGModel,
    "ChemBERTa": Chemberta,
}


class StepNotFoundError(Exception):
    """Exception raise for unsuccessful search of the best epoch/step for model training
    as implemented with model_Selection.get_best_steps_number.
    """
    def __init__(
            self,
            message="None of the steps shows statistically significant improvement"
    ):
        super().__init__(message)
