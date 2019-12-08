from .smallweightprune import SmallWeightPruner
from .smallweight_embedsplit import SmallWeightSplitEmbeddedPruner
from .smallweight_allsplit import SmallWeightSplitAll


PRUNERS = {
    SmallWeightPruner.code(): SmallWeightPruner,
    SmallWeightSplitEmbeddedPruner.code(): SmallWeightSplitEmbeddedPruner
    SmallWeightSplitAll.code(): SmallWeightSplitAll
}


def pruner_factory(args, model):
    pruner = PRUNERS[args.prune_code]
    return pruner(args, model)
