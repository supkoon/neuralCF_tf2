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
    parser = argparse.ArgumentParser(description="MLP.")
    parser.add_argument('--path', nargs='?', default='/dataset/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ratings.csv',
                        help='Choose a dataset.')
    parser.add_argument('--layers', nargs='+', default=[64,32,16,8],
                        help='num of layers and nodes of each layer. embedding size is (2/1st layer) ')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--regs', nargs='+', default=[0,0,0,0],
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

class MLP:

    def __init__(self,num_users,num_items,layers = [64,32,16,8] ,regs= [0,0,0,0]):
        self.num_users = num_users
        self.num_items = num_items
        self.layers = list(map(int,layers))
        self.num_layers = len(layers)
        self.regs = list(map(float,regs))
        #inputs
        user_input = keras.layers.Input(shape=(1,),dtype = 'int32')
        item_input = keras.layers.Input(shape=(1,),dtype = 'int32')

        #embedding layer : embedding_size = layer[0]/2
        user_embedding = keras.layers.Embedding(num_users,int(self.layers[0]/2),embeddings_regularizer=keras.regularizers.l2(self.regs[0]),
                                                name = "user_embedding")(user_input)
        item_embedding = keras.layers.Embedding(num_items,int(self.layers[0]/2),embeddings_regularizer=keras.regularizers.l2(self.regs[0]),
                                                name = 'item_embedding')(item_input)

        user_latent = keras.layers.Flatten()(user_embedding)
        item_latent = keras.layers.Flatten()(item_embedding)

        #concat  : layer 0 , size : layer[0]
        vector = keras.layers.concatenate([user_latent,item_latent])

        #hidden layers : 1 ~ num_layer
        for index in range(self.num_layers):
            layer = keras.layers.Dense(layers[index],kernel_regularizer=keras.regularizers.l2(self.regs[index]),
                                       activation = keras.activations.relu,
                                       name = f'layer{index}')
            vector =layer(vector)

        output = keras.layers.Dense(1,kernel_initializer=keras.initializers.lecun_uniform(),
                                    name='output'
                                    )(vector)

        self.model = keras.Model(inputs = [user_input,item_input],
                            outputs = [output])

    def get_model(self):
        model = self.model
        return model

if __name__ =="__main__":
        #argparse
        args = parse_args()
        layers = args.layers
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
        model = MLP(loader.num_users,loader.num_items,layers,regs).get_model()

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
        test_sample = X_test[:10]
        test_sample_label = test_labels[:10]
        print(model.predict([test_sample[:,0],test_sample[:,1]]))
        print(test_sample_label)

