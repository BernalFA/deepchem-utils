# DeepChem-utils

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

<br/>

Custom utility functions and classes for DL model exploration with [DeepChem](https://github.com/deepchem/deepchem). Most of them are wrappers to existing `DeepChem` objects.

## Features

1. Dataset preparation wrapper for datasets with user-defined splits.

```python
from deepchem_utils.dataset import prepare_dataset

train_dataset, valid_dataset, transformer = prepare_dataset(
    filepath=path_to_data,  # CSV file
    model_name="GraphConvModel",
    valid_indices=valid_indices,
    test_indices=test_indices
)
```

2. Hyperparameter optimization wrapper.

```python
import deepchem as dc
from deepchem_utils.model_selection import run_hyperopt_search

f1 = dc.metrics.Metric(dc.metrics.f1_score)

run_hyperopt_search(
    model_name="GraphConvModel",
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    params_dict=params_dict,
    metrics=f1,
    output_filepath="data",
    transformer=transformer
)
```

3. Evaluation of models with optimized hyperparameters using a custom callback to define best number of epochs (pseudo-manual early stopping selection as `deepchem` does not has an implementation for it).

```python
from deepchem_utils.model_selection import SelectEpochs

experiment = SelectEpochs(
    model_name="GraphConvModel",
    params=params,
    metrics=[f1],
    frequency=2,
    nb_epoch=100,
    output_file="data/GraphConvModel_valid_callback.csv"
)
experiment.repeated_evaluation(train_dataset, valid_dataset, transformer, n_times=5)
```

## License

The content of this repo is licensed under the [MIT license](./LICENSE) conditions.