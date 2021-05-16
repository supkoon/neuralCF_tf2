#GMF with activation


import tensorflow as tf
from tensorflow import keras

import argparse

#args
def parse_args():
    parser = argparse.ArgumentParser(description="GMF")
    parser.add_argument('--path',default="dataset/",help="dataset path")
    parser.add_argument('--dataset',default='movielens',help="dataset")
    parser.add_argument()


class GMF:
    def __init__(self,num_users,num_items,latent_fetures = 8):

        self.latent_fetures = latent_fetures
        user_input = keras.layers.Input(shape=(1,),)
        #user embedding layer




