#GMF with activation


import tensorflow as tf
from tensorflow import keras
from functions.dataloader import dataloader

class GMF:

    def __init__(self,num_users,num_items,latent_fetures = 8):
        self.num_users = num_users
        self.num_items = num_items
        self.latent_features = latent_fetures
        #inputs
        user_input = keras.layers.Input(shape=(1,),dtype = 'int32')
        item_input = keras.layers.Input(shape=(1,),dtype = 'int32')
        #embedding layer
        user_embedding = keras.layers.Embedding(num_users,latent_fetures)(user_input)
        item_embedding = keras.layers.Embedding(num_items,latent_fetures)(item_input)
        user_latent = keras.layers.Flatten()(user_embedding)
        item_latent = keras.layers.Flatten()(item_embedding)

        #concat with multiply
        concat = keras.layers.Multiply([user_latent,item_latent])

        output = keras.layers.Dense(1,kernel_initializer=keras.initializers.lecun_uniform,
                                    )(concat)

        self.model = keras.Model(inputs = [user_input,item_input],
                            outputs = [output])

        self.model.compile(optimizer=keras.optimizers.Adam(),
                           loss=keras.losses.binary_crossentropy)


    def get_model(self):
        model = self.model
        return model

if __name__ =="__main__":
        loader = dataloader("/Users/koosup/PycharmProjects/NCF/dataset/movielens")
        train_data = loader.train_data
        test_data = loader.test_data
        print(train_data.shape)
        print(test_data.shape)





