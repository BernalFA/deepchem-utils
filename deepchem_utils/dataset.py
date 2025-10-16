from pathlib import Path
from typing import Union

from deepchem.data import Dataset, CSVLoader
from deepchem.feat import Featurizer, create_char_to_idx
from deepchem.trans import Transformer, BalancingTransformer, DAGTransformer
from deepchem.splits import SpecifiedSplitter

from deepchem_utils.utils import featurizer4model


class Dataset:
    def __init__(self, filepath: Union[str, Path], feature_field: str, tasks: str,
                 featurizer: Featurizer, balancing: bool = True,
                 DAGtransform: bool = False):
        self.filepath = filepath
        self.featurizer = featurizer
        self.feature_field = feature_field
        self.tasks = tasks
        self.balancing = balancing
        self.DAGtransform = DAGtransform
        self.transformer = None

    def prepare_dataset(self):
        loader = CSVLoader(tasks=self.tasks, feature_field=self.feature_field,
                           featurizer=self.featurizer)
        dataset = loader.create_dataset(self.filepath)
        if self.DAGtransform:
            dataset = self._dag_transform(dataset)
        self.dataset = dataset

    def _bal_transform(self, dataset: Dataset) -> Dataset:
        transformer = BalancingTransformer(dataset=dataset)
        dataset = transformer.transform(dataset)
        return dataset

    def _dag_transform(self, dataset: Dataset) -> Dataset:
        transformer = DAGTransformer()
        dataset = transformer.transform(dataset=dataset)
        self.transformer = transformer
        return dataset

    def split_dataset(
            self, valid_indices: list, test_indices: list
    ) -> tuple[Dataset, Dataset, Dataset]:
        splitter = SpecifiedSplitter(
            valid_indices=valid_indices, test_indices=test_indices
        )
        train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
            self.dataset
        )
        if self.balancing:
            train_dataset = self._bal_transform(train_dataset)
        return train_dataset, valid_dataset, test_dataset


def prepare_dataset(
        filepath: Union[str, Path], model_name: str, valid_indices: list,
        test_indices: list
) -> tuple[Dataset, Dataset, Dataset, Transformer]:
    if model_name == "Smiles2Vec":
        char_to_idx = create_char_to_idx(filename=filepath, smiles_field="smiles")
        featurizer = featurizer4model(model_name, char_to_idx=char_to_idx)
    else:
        featurizer = featurizer4model(model_name)
    dataset = Dataset(
        filepath=filepath,
        feature_field="smiles",
        tasks=["active"],
        featurizer=featurizer,
        DAGtransform=True if model_name == "DAGModel" else False
    )
    dataset.prepare_dataset()
    train_dataset, valid_dataset, test_dataset = dataset.split_dataset(
                valid_indices=valid_indices, test_indices=test_indices
    )
    return train_dataset, valid_dataset, test_dataset, dataset.transformer
