""" Callback class modified from original deepchem callback
"""
from pathlib import Path

import pandas as pd


class CustomValidationCallback:
    """Edited from dc.models.callbacks.

    """

    def __init__(self,
                 valid_dataset,
                 interval,
                 metrics,
                 output_file=None,
                 save_dir=None,
                 save_metric=0,
                 save_on_minimum=True,
                 transformers=[]):
        """Create a ValidationCallback.

        Parameters
        ----------
        valid_dataset: dc.data.Dataset
            the validation set on which to compute the metrics
        interval: int
            the interval (in training steps) at which to perform validation
        metrics: list of dc.metrics.Metric
            metrics to compute on the validation set
        output_file: file
            to file to which results should be written
        save_dir: str
            if not None, the model parameters that produce the best validation score
            will be written to this directory
        save_metric: int
            the index of the metric to use when deciding whether to write a new set
            of parameters to disk
        save_on_minimum: bool
            if True, the best model is considered to be the one that minimizes the
            validation metric.  If False, the best model is considered to be the one
            that maximizes it.
        transformers: List[Transformer]
            List of `dc.trans.Transformer` objects. These transformations
            must have been applied to `dataset` previously. The dataset will
            be untransformed for metric evaluation.
        """
        self.valid_dataset = valid_dataset
        self.interval = interval
        self.metrics = metrics
        self.output_file = output_file
        self.save_dir = save_dir
        self.save_metric = save_metric
        self.save_on_minimum = save_on_minimum
        self._best_score = None
        self.transformers = transformers

    def __call__(self, model, step):
        if step % self.interval != 0:
            return None
        scores = model.evaluate(self.valid_dataset, self.metrics, self.transformers)
        scores["step"] = step
        self.results = scores

        if self.output_file is not None:
            self._scores_to_file(scores)

        if model.tensorboard:
            for key in scores:
                model._log_scalar_to_tensorboard(key, scores[key],
                                                 model.get_global_step())
        score = scores[self.metrics[self.save_metric].name]
        if not self.save_on_minimum:
            score = -score
        if self._best_score is None or score < self._best_score:
            self._best_score = score
            if self.save_dir is not None:
                model.save_checkpoint(model_dir=self.save_dir)
        if model.wandb_logger is not None:
            # Log data to Wandb
            data = {'eval/' + k: v for k, v in scores.items()}
            model.wandb_logger.log_data(data, step, dataset_id=id(self.dataset))

    def get_best_score(self):
        """This getter returns the best score evaluated on the validation set.

        Returns
        -------
        float
            The best score.
        """
        if self.save_on_minimum:
            return self._best_score
        else:
            return -self._best_score

    def _scores_to_file(self, scores):
        df = pd.DataFrame([scores])
        if Path.exists(self.output_file):
            df.to_csv(self.output_file, mode="a", header=False, index=False)
        else:
            df.to_csv(self.output_file, index=False)
