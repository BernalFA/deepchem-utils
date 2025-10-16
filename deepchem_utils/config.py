from deepchem.feat import (
    ConvMolFeaturizer, DMPNNFeaturizer, DummyFeaturizer, GroverFeaturizer,
    MolGraphConvFeaturizer, SmilesToSeq,
)
from deepchem.models import GraphConvModel, DAGModel, Smiles2Vec
from deepchem.models.torch_models import (
    MPNNModel, DMPNNModel, GCNModel, AttentiveFPModel, GATModel, Chemberta, GroverModel
)


FEATURIZERS = [
    ConvMolFeaturizer,
    DMPNNFeaturizer,
    DummyFeaturizer,
    GroverFeaturizer,
    MolGraphConvFeaturizer,
    SmilesToSeq,
]


MODELS = {
    "GraphConvModel": GraphConvModel,
    "MPNNModel": MPNNModel,
    "DMPNNModel": DMPNNModel,
    "GCNModel": GCNModel,
    "AttentiveFPModel": AttentiveFPModel,
    "GATModel": GATModel,
    "DAGModel": DAGModel,
    "ChemBERTa": Chemberta,
    "Smiles2Vec": Smiles2Vec,
    "GroverModel": GroverModel,
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
