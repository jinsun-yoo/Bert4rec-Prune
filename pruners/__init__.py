from .prune import SmallWeightPruner


PRUNERS = {
    SmallWeightPruner.code(): SmallWeightPruner,
}


def pruner_factory(args, model):
    pruner = PRUNERS[args.pruner_code]
    return pruner(args, model)
