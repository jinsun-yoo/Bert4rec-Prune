from .ml_1m import ML1MDataset
from .beauty import BeautyDataset
from .steam import SteamDataset

DATASETS = {
    ML1MDataset.code(): ML1MDataset,
    BeautyDataset.code(): BeautyDataset,
    SteamDataset.code(): SteamDataset,
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
