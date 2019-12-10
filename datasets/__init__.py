from .ml_1m import ML1MDataset
from .beauty import BeautyDataset
from .steam import SteamDataset
from .ml_20m import ML20MDataset

DATASETS = {
    ML1MDataset.code(): ML1MDataset,
    BeautyDataset.code(): BeautyDataset,
    SteamDataset.code(): SteamDataset,
    ML20MDataset.code(): ML20MDataset,
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
