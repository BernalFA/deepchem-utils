import math

from deepchem_utils.config import FEATURIZERS


def featurizer4model(model_name: str):
    if model_name in ["GraphConvModel", "DAGModel"]:
        featurizer = FEATURIZERS["ConvMolFeaturizer"]
    elif model_name in ["MPNNModel", "GATModel", "GCNModel", "AttentiveFPModel"]:
        featurizer = FEATURIZERS["MolGraphConvFeaturizer"]
    elif model_name == "DMPNNModel":
        featurizer = FEATURIZERS["DMPNNFeaturizer"]
    else:
        raise NotImplementedError(f"{model_name} not available")
    return featurizer


class IntervalEpochConv:
    def __init__(self, dataset_size: int, batch_size: int):
        self.dataset_size = dataset_size
        self.batch_size = batch_size

    def calculate_interval(self, frequency: int = 2):
        steps_per_epoch = math.ceil(self.dataset_size / self.batch_size)
        return steps_per_epoch / frequency

    def calculate_early_stopping_epoch(self, step: int):
        steps_per_epoch = math.ceil(self.dataset_size / self.batch_size)
        return round(step / steps_per_epoch, 2)
