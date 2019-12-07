from .smallweightprune import SmallWeightPruner, SmallWeightEmbeddedPruner


PRUNERS = {
    SmallWeightPruner.code(): SmallWeightPruner,
    SmallWeightSplitEmbeddedPruner.code(): SmallWeightSplitEmbeddedPruner
}


def pruner_factory(args, model):
    pruner = PRUNERS[args.prune_code]
    return pruner(args, model)
