from enum import Enum

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


class ModelsEnum(Enum):
    AttentiveFP = (AttentiveFPModel,
                   {"n_tasks": 1, "n_classes": 2})
    ChemBERTa = (Chemberta,
                 {"n_tasks": 1, "n_classes": 2, "max_length": 512,
                  "truncation": True, "padding": "max_length",
                  "tokenizer_path": "DeepChem/ChemBERTa-77M-MLM"})
    DAG = (DAGModel,
           {"n_tasks": 1, "n_classes": 2})
    DMPNN = (DMPNNModel,
             {"n_tasks": 1, "n_classes": 2})
    GAT = (GATModel,
           {"n_tasks": 1, "n_classes": 2})
    GCN = (GCNModel,
           {"n_tasks": 1, "n_classes": 2})
    GraphConv = (GraphConvModel,
                 {"n_tasks": 1, "n_classes": 2})
    Grover = (GroverModel,
              {"n_tasks": 1, "n_classes": 2, "task": "finetuning",
               "features_dim": 2048, "hidden_size": 128,
               "functional_group_size": 85})
    MPNN = (MPNNModel,
            {"n_tasks": 1, "n_classes": 2})
    Smiles2Vec = (Smiles2Vec,
                  {"n_tasks": 1, "n_classes": 2})

    @property
    def cls(self):
        return self.value[0]

    @property
    def defaults(self):
        return self.value[1]


class StepNotFoundError(Exception):
    """Exception raise for unsuccessful search of the best epoch/step for model training
    as implemented with model_Selection.get_best_steps_number.
    """
    def __init__(
            self,
            message="None of the steps shows statistically significant improvement"
    ):
        super().__init__(message)
