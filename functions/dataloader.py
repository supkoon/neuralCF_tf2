import pandas as pd
import os

class dataloader():

    def __init(self):
        pass
    def load_dataset(self):
        path = "/Users/koosup/PycharmProjects/NCF/dataset/movielens"
        # path = os.curdir+ "/NCF/dataset/movielens"
        ratings_df = pd.read_csv(os.path.join(path, 'ratings.csv'), encoding='utf-8')
        movies_df = pd.read_csv(os.path.join(path, 'movies.csv'), index_col='movieId', encoding='utf-8')
        tags_df = pd.read_csv(os.path.join(path, 'tags.csv'), encoding='utf-8')

print(os.curdir)