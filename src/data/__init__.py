from importlib import import_module


def get_dataloder(args):
    module = import_module('data.' + args.task)
    return module.MyDataLoader(args)