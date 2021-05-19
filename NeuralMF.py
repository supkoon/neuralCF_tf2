#GMF with activation
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from functions.dataloader import dataloader
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from GMF import GMF
from MLP import MLP
import numpy as np

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="NeuralMF.")
    parser.add_argument('--path', nargs='?', default='/dataset/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ratings.csv',
                        help='Choose a dataset.')
    parser.add_argument('--layers', nargs='+', default=[64,32,16,8],
                        help='num of layers and nodes of each layer. embedding size is (2/1st layer) ')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size of MF model.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--gmf_regs', type=float, default=0,
                        help='Regularization for MF embeddings.')
    parser.add_argument('--mlp_regs', nargs='+', default=[0,0,0,0],
                        help="Regularization for user and item embeddings.")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.(1 or 0)')
    parser.add_argument('--patience', type=int, default=10,
                        help='earlystopping patience')
    parser.add_argument('--pretrain_gmf', nargs='?', default='',
                        help='')
    parser.add_argument('--pretrain_mlp', nargs='?', default='',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='pretrain trade off between GMF:alpha || MLP:1-alpha ')
    return parser.parse_args()

class NeuralMF:

    def __init__(self,num_users,num_items,latent_features=8,layers = [64,32,16,8] ,gmf_regs=0,mlp_regs= [0,0,0,0]):
        self.num_users = num_users
        self.num_items = num_items
        #gmf
        self.latent_features = latent_features
        self.gmf_regs = gmf_regs
        #mlp
        self.layers = list(map(int,layers))
        self.num_layers = len(layers)
        self.mlp_regs = list(map(float,mlp_regs))

        #inputs
        user_input = keras.layers.Input(shape=(1,),dtype = 'int32')
        item_input = keras.layers.Input(shape=(1,),dtype = 'int32')

        #GMF_embedding_layer : embedding_size = num_factors
        user_embedding_gmf = keras.layers.Embedding(num_users, self.latent_features,
                                                embeddings_regularizer=keras.regularizers.l2(self.gmf_regs),
                                                    name = 'user_embedding_gmf')(user_input)
        item_embedding_gmf = keras.layers.Embedding(num_items, self.latent_features,
                                                embeddings_regularizer=keras.regularizers.l2(self.gmf_regs),
                                                    name = 'item_embedding_gmf')(item_input)
        user_latent_gmf = keras.layers.Flatten()(user_embedding_gmf)
        item_latent_gmf = keras.layers.Flatten()(item_embedding_gmf)

        result_gmf = keras.layers.Multiply()([user_latent_gmf,item_latent_gmf])

        #mlp_embedding layer : embedding_size = layer[0]/2
        user_embedding_mlp = keras.layers.Embedding(num_users,int(self.layers[0]/2),embeddings_regularizer=keras.regularizers.l2(self.mlp_regs[0]),
                                                    name = 'user_embedding_mlp')(user_input)
        item_embedding_mlp = keras.layers.Embedding(num_items,int(self.layers[0]/2),embeddings_regularizer=keras.regularizers.l2(self.mlp_regs[0]),
                                                    name = 'item_embedding_mlp')(item_input)

        user_latent_mlp = keras.layers.Flatten()(user_embedding_mlp)
        item_latent_mlp = keras.layers.Flatten()(item_embedding_mlp)

        result_mlp = keras.layers.concatenate([user_latent_mlp,item_latent_mlp])

        #mlp hidden layers : 1 ~ num_layer
        for index in range(self.num_layers):
            layer = keras.layers.Dense(layers[index],kernel_regularizer=keras.regularizers.l2(self.mlp_regs[index]),
                                       activation = keras.activations.relu,
                                       name = f'layer{index}')
            result_mlp =layer(result_mlp)


        #concat (gmf_result, mlp result)
        concat = keras.layers.concatenate([result_gmf,result_mlp])

        #predict rating
        output = keras.layers.Dense(1,kernel_initializer=keras.initializers.lecun_uniform(),
                                    name='output'
                                    )(concat)

        self.model = keras.Model(inputs = [user_input,item_input],
                            outputs = [output])


    def get_model(self):
        model = self.model
        return model


def load_pretrain_model(model, gmf_model, mlp_model, num_layers,alpha):
    # MF embeddings
    gmf_user_embeddings = gmf_model.get_layer('user_embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('item_embedding').get_weights()
    model.get_layer('user_embedding_gmf').set_weights(gmf_user_embeddings)
    model.get_layer('item_embedding_gmf').set_weights(gmf_item_embeddings)

    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
    model.get_layer('user_embedding_mlp').set_weights(mlp_user_embeddings)
    model.get_layer('item_embedding_mlp').set_weights(mlp_item_embeddings)

    # MLP layers
    for i in range(num_layers):
        mlp_layer_weights = mlp_model.get_layer(f'layer{i}').get_weights()

        model.get_layer(f'layer{i}').set_weights(mlp_layer_weights)


    # Prediction weights with hyper parameter 'alpha'
    gmf_output = gmf_model.get_layer('output').get_weights()
    mlp_output = mlp_model.get_layer('output').get_weights()


    pretrain_weights = np.concatenate((alpha * gmf_output[0], (1-alpha)*mlp_output[0]), axis=0)
    pretrain_bias = alpha * gmf_output[1] + (1-alpha)*alpha* mlp_output[1]
    model.get_layer('output').set_weights([pretrain_weights, pretrain_bias])
    return model


if __name__ =="__main__":
        #argparse
        args = parse_args()
        layers = args.layers
        num_factors = args.num_factors
        mlp_regs = args.mlp_regs
        gmf_regs = args.gmf_regs
        learner = args.learner
        learning_rate = args.lr
        epochs = args.epochs
        batch_size = args.batch_size
        patience = args.patience
        pretrain_gmf = args.pretrain_gmf
        pretrain_mlp = args.pretrain_mlp
        alpha = args.alpha

        #load datasets
        loader = dataloader(args.path + args.dataset)
        X_train,labels = loader.generate_trainset()
        X_test,test_labels =loader.generate_testset()

        #callbacks
        early_stop_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        model_out_file = 'Pretrain/NeuralMF_%s.h5' % (datetime.now().strftime('%Y-%m-%d-%h-%m-%s'))
        model_check_cb = keras.callbacks.ModelCheckpoint(model_out_file, save_best_only=True)

        #model
        model = NeuralMF(loader.num_users,loader.num_items,num_factors,layers,gmf_regs,mlp_regs).get_model()

        if pretrain_gmf != '' and pretrain_mlp != '':
            gmf_model = GMF(loader.num_users, loader.num_items,num_factors,gmf_regs).get_model()
            gmf_model.load_weights(pretrain_gmf)
            mlp_model = MLP(loader.num_users, loader.num_items, layers, mlp_regs).get_model()
            mlp_model.load_weights(pretrain_mlp)
            model = load_pretrain_model(model, gmf_model, mlp_model, len(layers), alpha =alpha)
            print(f"Load pretrained GMF ({pretrain_gmf}) and MLP ({pretrain_mlp}) models done. ")



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

