# Neural Collaborative Filtering with tensorflow 2


## --reference paper

Generalized Matrix Factorization (GMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.  


## --files
+ dataset : Movielens
+ GMF : predict 0~5 ratings, not [0 or 1] interaction
+ MLP : predict 0~5 ratings, not [0 or 1] interaction
+ NeuralCF :

## example : GMF
```
python GMF.py --path "/Users/koosup/PycharmProjects/NCF/dataset/movielens/" --dataset "ratings.csv" --epochs 10 --batch_size 32 --num_factors 8 --regs [0,0] --lr 0.001 --learner adam --out 1 --patience 10
```
## example : MLP
'''
python MLP.py --path "/Users/koosup/PycharmProjects/NCF/dataset/movielens/" --dataset "ratings.csv" --epochs 20 --batch_size 256 --layers [64,32,16,8] --reg_layers [0,0,0,0] --lr 0.001 --learner adam --out 1 --patience 10
'''
