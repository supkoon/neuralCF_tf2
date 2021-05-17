#GMF with activation


import tensorflow as tf
from tensorflow import keras
from functions.dataloader import dataloader
import numpy as np
import matplotlib.pyplot as plt
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
        concat = keras.layers.Multiply()([user_latent,item_latent])

        output = keras.layers.Dense(1,kernel_initializer=keras.initializers.lecun_uniform,
                                    )(concat)

        self.model = keras.Model(inputs = [user_input,item_input],
                            outputs = [output])

        self.model.compile(optimizer=keras.optimizers.Adam(),
                           loss=keras.losses.mean_squared_error,
                           )


    def get_model(self):
        model = self.model
        return model

if __name__ =="__main__":
        loader = dataloader("/Users/koosup/PycharmProjects/NCF/dataset/movielens")

        X_train,labels = loader.generate_trainset()
        X_test,test_labels =loader.generate_testset()

        model = GMF(loader.num_users,loader.num_movies).get_model()

        history = model.fit([X_train[:,0],X_train[:,1]],labels,
                  epochs=10,
                  batch_size=32,
                  validation_data=([X_test[:,0],X_test[:,1]],test_labels)
                  )

        plt.plot(history.history)




