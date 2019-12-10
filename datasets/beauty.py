from .base import AbstractDataset

import pandas as pd

from datetime import date


class BeautyDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'beauty'

    @classmethod
    def url(cls):
        return 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['Beauty_5.json']

    @classmethod
    def raw_filetype(cls):
        return 'gz'

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('Beauty_5.json')
        df = pd.read_json(file_path, lines=True)
        df.columns = ['uid', 'sid', 'uname', 'helpful', 'review', 'rating', 'summary', 'timestamp', 'datetime']
        df = df.drop(columns=['uname', 'helpful', 'review', 'summary', 'datetime'])
        return df
