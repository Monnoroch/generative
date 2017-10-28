"""
Tools for managing experiments.
"""

import inspect
import json
import os


class Experiment(object):
    def __init__(self, name):
        self.name = name

    @staticmethod
    def from_checkpoint(checkpoint):
        return Experiment(os.path.dirname(os.path.dirname(os.path.dirname(checkpoint))))

    def experiment_dir(self):
        _create_if_not_existis(self.name)
        return self.name

    def summaries_dir(self):
        summaries_dir = os.path.join(self.experiment_dir(), "summaries")
        _create_if_not_existis(summaries_dir)
        return summaries_dir

    def checkpoint(self, step):
        return os.path.join(self._train_dir(), "checkpoint-%d" % step, "data")

    def _train_dir(self):
        train_dir = os.path.join(self.experiment_dir(), "model")
        _create_if_not_existis(train_dir)
        return train_dir

    def _hparams_file_path(self):
        return os.path.join(self.experiment_dir(), "hparams.json")

    def load_hparams(self, hparams_class, args=None):
        hparams_file = self._hparams_file_path()
        if not os.path.exists(hparams_file):
            result = hparams_class(args)
            self._save_hparams(result)
            return result
        with open(hparams_file, "r") as file:
            hparams = json.load(file)

        class Hparams(object):
            def __init__(self, hparams):
                for key, value in hparams.items():
                   setattr(self, key, value)
        return hparams_class(Hparams(hparams))

    def _save_hparams(self, hparams):
        props = {}
        for name in dir(hparams):
            value = getattr(hparams, name)
            if not name.startswith('__') and not inspect.ismethod(value):
                props[name] = value
        with open(self._hparams_file_path(), "w") as file:
            json.dump(props, file, indent=2)
            file.write("\n")

def _create_if_not_existis(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


