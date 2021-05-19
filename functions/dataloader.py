import pandas as pd
import os

import numpy as np


from sklearn.model_selection import train_test_split

class dataloader():

    def __init__(self,data_path):
        self.data_path = data_path
        ratings_df = pd.read_csv(os.path.join(self.data_path), encoding='utf-8')
        self.train_df, self.test_df = train_test_split(ratings_df, test_size=0.2, random_state=42, shuffle=True)
        # user dataset
        self.users = self.train_df["userId"].unique()
        self.num_users = len(self.users)
        self.user_to_index = {user : idx for idx,user in enumerate(self.users)}
        # movie dataset
        self.movies = self.train_df["movieId"].unique()
        self.num_items = len(self.movies)

        self.movies_to_index = {movie : idx for idx, movie in enumerate(self.movies)}

        #test_df user,movie must be in train_df
        self.test_df = self.test_df[self.test_df["userId"].isin(self.users) & self.test_df["movieId"].isin(self.movies)]

        # target dataset
    def generate_trainset(self):

        X_train = pd.DataFrame({'user' : self.train_df["userId"].map(self.user_to_index),
                                    'movie': self.train_df["movieId"].map(self.movies_to_index)})
        y_train = self.train_df["rating"].astype(np.float32)
        return np.asarray(X_train), np.asarray(y_train)

    def generate_testset(self):
        X_test = pd.DataFrame({'user' : self.test_df["userId"].map(self.user_to_index),
                               "movie": self.test_df["movieId"].map(self.movies_to_index)})
        y_test = self.test_df['rating'].astype(np.float32)
        return np.asarray(X_test),np.asarray(y_test)
# if __name__ == "__main__":
#     loader = dataloader("/Users/koosup/PycharmProjects/NCF/dataset/movielens")
#     train_x,train_y = loader.generate_trainset()
#     test_x,test_y = loader.generate_testset()
#     print(train_x)
#     print(train_y)
#     print(len(train_x))
#     print(test_x)
#     print(test_y)
#     print(len(test_x))













