# Neural Collaborative Filtering with tensorflow 2


## --reference paper

Generalized Matrix Factorization (GMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.  


## --files
+ dataset : Movielens
+ GMF : predict 0~5 ratings, not [0 or 1] interaction
+ MLP : predict 0~5 ratings, not [0 or 1] interaction
+ NeuralCF : predict 0~5 ratiings, not [0 or 1] interaction, can use pretrain trade-off between GMF, MLP wih hyper parameter alpha 

## example : GMF
```
python GMF.py --path "/Users/koosup/PycharmProjects/NCF/dataset/movielens/" --dataset "ratings.csv" --epochs 10 --batch_size 32 --num_factors 8 --regs [0,0] --lr 0.001 --learner adam --out 1 --patience 10
```
## example : MLP
```
python MLP.py --path "/Users/koosup/PycharmProjects/NCF/dataset/movielens/" --dataset "ratings.csv" --epochs 20 --batch_size 256 --layers [64,32,16,8] --reg_layers [0,0,0,0] --lr 0.001 --learner adam --out 1 --patience 10
```
## example : NeuralMF
```
python NeuralMF --path "/Users/koosup/PycharmProjects/NCF/dataset/movielens/" --dataset "ratings.csv" --epochs 20 --batch_size 256 --layers [64,32,16,8] --num_factors 8 --gmf_regs 0 --mlp_regs [0,0,0,0] --pretrain_gmf '/Users/koosup/PycharmProjects/NCF/Pretrain/GMF_2021-05-20-May-05-1621448652.h5' --pretrain_mlp '/Users/koosup/PycharmProjects/NCF/Pretrain/MLP_2021-05-20-May-05-1621450141.h5' --alpha 0.3
```
