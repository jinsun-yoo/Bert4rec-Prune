from .smallweightprune import Pruner_Linear
from .smallweight_embedsplit import Pruner_Linear_Embed
from .smallweight_allsplit import PrunerFinal

PRUNERS = {
    PrunerFinal.code(): PrunerFinal,
    Pruner_Linear_Embed.code(): Pruner_Linear_Embed,
    Pruner_Linear.code(): Pruner_Linear
}


def pruner_factory(args, model):
    pruner = PRUNERS[args.prune_code]
    return pruner(args, model)
