from .smallweightprune import SmallWeightPruner
from .smallweight_embedsplit import SmallWeightSplitEmbeddedPruner


PRUNERS = {
    SmallWeightPruner.code(): SmallWeightPruner,
    SmallWeightSplitEmbeddedPruner.code(): SmallWeightSplitEmbeddedPruner
}


def pruner_factory(args, model):
    pruner = PRUNERS[args.prune_code]
    return pruner(args, model)
