import inspect
import json
import os


def _hparams_file_path(args):
    return os.path.join(args.experiment_dir, "hparams.json")


def load_hparams(args):
    hparams_file = _hparams_file_path(args)
    if not os.path.exists(hparams_file):
        return args
    with open(hparams_file, "r") as file:
        hparams = json.load(file)

    class Hparams(object):
        def __init__(self, hparams):
            for key, value in hparams.items():
               setattr(self, key, value)
    return Hparams(hparams)


def save_hparams(hparams, args):
    props = {}
    for name in dir(hparams):
        value = getattr(hparams, name)
        if not name.startswith('__') and not inspect.ismethod(value):
            props[name] = value
    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)
    with open(_hparams_file_path(args), "w") as file:
        json.dump(props, file, indent=2)
        file.write("\n")
