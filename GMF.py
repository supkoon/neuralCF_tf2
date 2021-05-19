#GMF with activation
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from functions.dataloader import dataloader
import matplotlib.pyplot as plt
import pandas as pd
import argparse

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="GMF.")
    parser.add_argument('--path', nargs='?', default='/dataset/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ratings.csv',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[0,0]',
                        help="Regularization for user and item embeddings.")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.(1 or 0)')
    parser.add_argument('--patience', type=int, default=10,
                        help='earlystopping patience')
    return parser.parse_args()

class GMF:

    def __init__(self,num_users,num_items,latent_fetures = 8,regs=[0,0]):
        self.num_users = num_users
        self.num_items = num_items
        self.latent_features = latent_fetures
        self.regs = regs
        #inputs
        user_input = keras.layers.Input(shape=(1,),dtype = 'int32')
        item_input = keras.layers.Input(shape=(1,),dtype = 'int32')
        #embedding layer
        user_embedding = keras.layers.Embedding(num_users,latent_fetures,embeddings_regularizer=keras.regularizers.l2(self.regs[0]))(user_input)
        item_embedding = keras.layers.Embedding(num_items,latent_fetures,embeddings_regularizer=keras.regularizers.l2(self.regs[1]))(item_input)
        user_latent = keras.layers.Flatten()(user_embedding)
        item_latent = keras.layers.Flatten()(item_embedding)

        #concat with multiply
        concat = keras.layers.Multiply()([user_latent,item_latent])

        output = keras.layers.Dense(1,kernel_initializer=keras.initializers.lecun_uniform(),
                                    )(concat)

        self.model = keras.Model(inputs = [user_input,item_input],
                            outputs = [output])

    def get_model(self):
        model = self.model
        return model

if __name__ =="__main__":
        #argparse
        args = parse_args()
        num_factors = args.num_factors
        regs = args.regs
        learner = args.learner
        learning_rate = args.lr
        epochs = args.epochs
        batch_size = args.batch_size
        patience = args.patience

        #load datasets
        loader = dataloader(args.path + args.dataset)
        X_train,labels = loader.generate_trainset()
        X_test,test_labels =loader.generate_testset()


        #callbacks
        early_stop_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        model_out_file = 'Pretrain/MLP_%s.h5' % (datetime.now().strftime('%Y-%m-%d-%h-%m-%s'))
        model_check_cb = keras.callbacks.ModelCheckpoint(model_out_file, save_best_only=True)

        #model
        model = GMF(loader.num_users,loader.num_movies,num_factors,regs).get_model()

        if learner.lower() == "adagrad":
            model.compile(optimizer=keras.optimizers.Adagrad(lr=learning_rate), loss='mse')
        elif learner.lower() == "rmsprop":
            model.compile(optimizer=keras.optimizers.RMSprop(lr=learning_rate), loss='mse')
        elif learner.lower() == "adam":
            model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss='mse')
        else:
            model.compile(optimizer=keras.optimizers.SGD(lr=learning_rate), loss='mse')
        #train
        if args.out:
            history = model.fit([X_train[:,0],X_train[:,1]],labels,
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_data=([X_test[:,0],X_test[:,1]],test_labels),
                                callbacks=[early_stop_cb,
                                           model_check_cb]
                      )
        else :
            history = model.fit([X_train[:, 0], X_train[:, 1]], labels,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_data=([X_test[:, 0], X_test[:, 1]], test_labels),
                                callbacks=[early_stop_cb]
                                )

        pd.DataFrame(history.history).plot(figsize= (8,5))
        plt.show()



