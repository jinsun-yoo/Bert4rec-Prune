from .smallweightprune import SmallWeightPruner


PRUNERS = {
    SmallWeightPruner.code(): SmallWeightPruner,
}


def pruner_factory(args, model):
    pruner = PRUNERS[args.prune_code]
    return pruner(args, model)
