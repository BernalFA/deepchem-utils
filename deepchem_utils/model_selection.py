import time
import json
from pathlib import Path

import deepchem as dc
import pandas as pd
from tqdm import tqdm

from deepchem_utils.callback import CustomValidationCallback
from deepchem_utils.config import MODELS
from deepchem_utils.utils import IntervalEpochConv


def run_hyperopt_search(model_name, train_dataset, valid_dataset, params_dict, metrics,
                        output_filepath, transformer):
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


def get_best_steps_number(df, patience=3):
    tmp = df.groupby(by="step").agg("mean")

    best_steps = []
    for col in tmp.columns:
        wait = 0
        best_val = 0
        best_id = None
        for i, row in tmp.iterrows():
            if wait < patience:
                if row[col] > best_val:
                    best_val = row[col]
                    best_id = i
                    wait = 0
                else:
                    wait += 1
            else:
                break

        best_steps.append(best_id)

    return tmp.loc[best_steps]


class EvaluateBestModels:
    def __init__(self, model_name, params, metrics, frequency, nb_epoch, output_file):
        self.model_name = model_name
        self.params = params
        self.metrics = metrics
        self.frequency = frequency
        self.nb_epoch = nb_epoch
        self.output_file = output_file

    def repeated_evaluation(
        self, train_dataset, valid_dataset, transformer=[], n_times=5
    ):
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
            epochs = self._select_early_stop()
        return epochs

    def evaluation(self, train_dataset, valid_dataset, transformer=[]):
        self._set_converter(train_dataset)
        callback = self._set_callback(
            valid_dataset=valid_dataset,
            interval=self._converter.calculate_interval(self.frequency),
            transformer=transformer
        )
        model = MODELS[self.model_name](**self.params)
        model.fit(train_dataset, nb_epoch=self.nb_epoch, callbacks=callback)
        if Path.exists(self.output_file):
            epochs = self._select_early_stop()
        return epochs

    def _set_callback(self, valid_dataset, interval, transformer):
        callback = CustomValidationCallback(
            valid_dataset, interval=interval, metrics=self.metrics,
            output_file=self.output_file,
            save_on_minimum=False
        )
        return callback

    def _set_converter(self, train_dataset):
        self._converter = IntervalEpochConv(
            train_dataset.X.shape[0], self.params["batch_size"]
        )
        return None

    def _select_early_stop(self):
        res = pd.read_csv(self.output_file)
        best = get_best_steps_number(res)
        epochs = []
        for step in best.index:
            epochs.append(self._converter.calculate_early_stopping_epoch(step))
        return epochs
