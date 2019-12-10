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
        return 'http://deepx.ucsd.edu/public/jmcauley/steam/australian_user_reviews.json.gz'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['australian_user_reviews.json']

    @classmethod
    def raw_filetype(cls):
        return 'gz'

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('australian_user_reviews.json')
        print(file_path)
        #df = pd.read_json(file_path, lines=True)
        #df.columns = ['uid', 'uurl', 'reviews']

        with open(file_path) as f:
            data = json.load(f)
        print(data)
        df= pd.DataFrame(data)
        print(df)
        #df = df.drop(columns=['uname', 'helpful', 'review', 'summary', 'datetime'])
        return df
