from .base import AbstractDataset

import pandas as pd
import json

from datetime import date


class SteamDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'steam'

    @classmethod
    def url(cls):
        return 'https://raw.githubusercontent.com/FeiSun/BERT4Rec/master/data/steam.txt'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['ratings.txt',]

    @classmethod
    def raw_filetype(cls):
        return 'txt'

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('ratings.txt')
        df = pd.read_csv(file_path, sep=' ', header=None)
        df.columns = ['uid', 'sid']
        df['rating'] = 1
        df['timestamp'] = df.index
        return df
