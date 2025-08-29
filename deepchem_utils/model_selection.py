import time
import json
from pathlib import Path
from typing import Union, Optional

import deepchem as dc
import pandas as pd
from scipy import stats
from tqdm import tqdm

from deepchem_utils.callback import CustomValidationCallback
from deepchem_utils.config import MODELS, StepNotFoundError
from deepchem_utils.utils import IntervalEpochConv


def run_hyperopt_search(model_name: str, train_dataset: dc.data.Dataset,
                        valid_dataset: dc.data.Dataset, params_dict: dict,
                        metrics: dc.metrics.Metric,
                        output_filepath: Union[str, Path],
                        transformer: Union[dc.trans.Transformer, list]):
    start = time.time()
    optimizer = dc.hyper.GridHyperparamOpt(MODELS[model_name])
    best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
        params_dict, train_dataset, valid_dataset, metrics,
        output_transformers=transformer
    )
    end = time.time()

    print("####" + "#" * len(model_name))
    print(f"# {model_name} #")
    print("####" + "#" * len(model_name))
    print(f'Total time required: {time.strftime("%H:%M:%S", time.gmtime(end - start))}')
    print(best_hyperparams)
    print(f"Best value: {max(all_results.values())}")
    if output_filepath:
        if isinstance(output_filepath, str):
            output_filepath = Path(output_filepath)
        output = output_filepath / f"{model_name}_best_hyperparameters.json"
        with open(output, "w") as file:
            json.dump(best_hyperparams, file, indent=4)


def get_best_steps_number(df: pd.DataFrame, alpha: int = 0.05, patience: int = 3,
                          max_step_fraction: float = 0.85) -> Optional[int]:
    # Define column for metrics and group dataframe
    col = [col for col in df.columns if col != "step"][0]
    grouped = df.groupby("step")[col]
    # Define keys (steps), best values as initial values
    keys = list(grouped.groups.keys())
    best_step = keys[0]
    best_val = grouped.get_group(best_step).mean()

    # Iterate over groups to find values whose difference is statistically significant
    selected_steps = []
    for step, group in grouped:
        best_vals = grouped.get_group(best_step)
        current_mean = group.mean()
        improvement = False
        # check data has variance (avoid adding useless data)
        if group.var() != 0 or best_vals.var() != 0:
            _, p = stats.ttest_ind(group, best_vals, equal_var=False)
            improvement = (current_mean > best_val) and (p < alpha)
        # update best values and store selected step
        if improvement:
            best_val = current_mean
            best_step = step
            selected_steps.append(step)

    # Iterate over selected steps and check differences and patience
    filtered = []
    for i in range(len(selected_steps) - 1):
        idx1 = selected_steps[i]
        idx2 = selected_steps[i + 1]
        _, p = stats.ttest_ind(grouped.get_group(idx1), grouped.get_group(idx2),
                               equal_var=False)
        if (p < alpha) and (idx2 - idx1 > patience):
            filtered.append(idx2)

    # Define best value as maximum if reached before 85% of the validation process
    if filtered:
        overall_best_step = max(filtered)
        top = keys[-1] - keys[-1] * (1 - max_step_fraction)
        # print(selected, top)
        if overall_best_step > top:
            overall_best_step = max([i for i in filtered if i != overall_best_step])
        return overall_best_step
    return None


class SelectEpochs:
    def __init__(self, model_name: str, params: dict, metrics: list, frequency: int,
                 nb_epoch: int, output_file: Union[str, Path]):
        self.model_name = model_name
        self.params = params
        self.metrics = metrics
        self.frequency = frequency
        self.nb_epoch = nb_epoch
        self.output_file = output_file

    def repeated_evaluation(
        self, train_dataset: dc.data.Dataset, valid_dataset: dc.data.Dataset,
        transformer: Union[dc.trans.Transformer, list] = [], n_times: int = 5
    ) -> Union[int, str]:
        self._set_converter(train_dataset)
        callback = self._set_callback(
            valid_dataset=valid_dataset,
            interval=self._converter.calculate_interval(self.frequency),
            transformer=transformer
        )
        for _ in tqdm(range(n_times)):
            model = MODELS[self.model_name](**self.params)
            model.fit(train_dataset, nb_epoch=self.nb_epoch, callbacks=callback)
        if Path.exists(self.output_file):
            try:
                epochs = self._select_early_stop()
            except Exception as e:
                epochs = e
        return epochs

    def evaluation(
            self, train_dataset: dc.data.Dataset, valid_dataset: dc.data.Dataset,
            transformer: Union[dc.trans.Transformer, list] = []
    ) -> Union[int, str]:
        self._set_converter(train_dataset)
        callback = self._set_callback(
            valid_dataset=valid_dataset,
            interval=self._converter.calculate_interval(self.frequency),
            transformer=transformer
        )
        model = MODELS[self.model_name](**self.params)
        model.fit(train_dataset, nb_epoch=self.nb_epoch, callbacks=callback)
        if Path.exists(self.output_file):
            try:
                epochs = self._select_early_stop()
            except Exception as e:
                epochs = e
        return epochs

    def _set_callback(
            self, valid_dataset: dc.data.Dataset, interval: int,
            transformer: Union[dc.trans.Transformer, list]
    ) -> CustomValidationCallback:
        callback = CustomValidationCallback(
            valid_dataset, interval=interval, metrics=self.metrics,
            output_file=self.output_file,
            transformers=transformer,
            save_on_minimum=False
        )
        return callback

    def _set_converter(self, train_dataset: dc.data.Dataset):
        self._converter = IntervalEpochConv(
            train_dataset.X.shape[0], self.params["batch_size"]
        )
        return None

    def _select_early_stop(self) -> Union[int, str]:
        res = pd.read_csv(self.output_file)
        best = get_best_steps_number(res)
        # check an actual value was found
        if best is not None:
            epochs = self._converter.calculate_early_stopping_epoch(best)
        else:
            raise StepNotFoundError
        return epochs

    def select_early_stop_from_data(
            self, train_dataset: dc.data.Dataset
    ) -> Union[int, str]:
        self._set_converter(train_dataset)
        epochs = self._select_early_stop()
        return epochs
