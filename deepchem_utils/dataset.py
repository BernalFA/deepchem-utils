import deepchem as dc

from deepchem_utils.utils import featurizer4model


class Dataset:
    def __init__(self, filepath, feature_field, tasks, featurizer, balancing=True,
                 DAGtransform=False):
        self.filepath = filepath
        self.featurizer = featurizer
        self.feature_field = feature_field
        self.tasks = tasks
        self.balancing = balancing
        self.DAGtransform = DAGtransform
        self.transformer = None

    def prepare_dataset(self):
        loader = dc.data.CSVLoader(tasks=self.tasks, feature_field=self.feature_field,
                                   featurizer=self.featurizer)
        dataset = loader.create_dataset(self.filepath)
        if self.DAGtransform:
            dataset = self._dag_transform(dataset)
        self.dataset = dataset

    def _bal_transform(self, dataset):
        transformer = dc.trans.BalancingTransformer(dataset=dataset)
        dataset = transformer.transform(dataset)
        return dataset

    def _dag_transform(self, dataset):
        transformer = dc.trans.DAGTransformer()
        dataset = transformer.transform(dataset=dataset)
        self.transformer = transformer
        return dataset

    def split_dataset(self, valid_indices, test_indices):
        splitter = dc.splits.SpecifiedSplitter(
            valid_indices=valid_indices, test_indices=test_indices
        )
        train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
            self.dataset
        )
        if self.balancing:
            train_dataset = self._bal_transform(train_dataset)
        return train_dataset, valid_dataset, test_dataset


# get dataset
def prepare_dataset(filepath, model_name, valid_indices, test_indices):
    dataset = Dataset(
        filepath=filepath,
        feature_field="smiles",
        tasks=["active"],
        featurizer=featurizer4model(model_name),
        DAGtransform=True if model_name == "DAGModel" else False
    )
    dataset.prepare_dataset()
    train_dataset, valid_dataset, _ = dataset.split_dataset(
                valid_indices=valid_indices, test_indices=test_indices
    )
    return train_dataset, valid_dataset, dataset.transformer
