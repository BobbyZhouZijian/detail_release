import os
import io
# import PIL.Image, PIL.ImageDraw, PIL.ImageFont
import base64
import zipfile
import json
import requests
import matplotlib.pylab as pl
import numpy as np
import glob
import requests
import random as pyrandom
from concurrent import futures
from functools import partial
from scipy.ndimage import rotate
from IPython.display import Image, HTML, clear_output
from tqdm import tqdm_notebook, tnrange
import time
from typing import Any, MutableMapping, NamedTuple, Tuple
import jax
from jax import grad, jit, vmap
import jax.numpy as jnp

import haiku as hk
import math

from ml_collections import config_dict
import matplotlib.pylab as pl
import matplotlib.colors as mcolors
colors = pl.colormaps['Dark2'] 

from src.transformer import Transformer
from src.data import create_circ_cls_data, create_weights
from src.config import config
from src.train import *

from torchvision.datasets import MNIST
import numpy as np
import jax.numpy as jnp
from jax.tree_util import tree_map
from torch.utils import data
from torchvision.datasets import MNIST

from utils import save_weights

import argparse

#@title Config
num_seeds = 1 #@param {type:"integer"}
num_classes = 2
total_classes = 5
bs = 10
save_dir = 'results/mnist_data_attr'
os.makedirs(save_dir, exist_ok=True)

from datetime import datetime
now = datetime.now()
config.local_usage = True
config.size_distract = 0
config.training_steps = 5000
config.training_steps_gd = 5000
config.use_softmax = True
config.mnist_task = True

####
config.deq = True
config.gd_deq = True
####
config.pre_train_gd = False
config.train_gd_whitening = False
config.train_gd_lr = True
####

config.layer_norm = True
config.out_proj = True
config.in_proj = False
config.adam = True

config.output_size = total_classes

config.dataset_size = bs - 1
config.input_size = 39
config.key_size = 40 #config.input_size + 1
config.num_layers = 10
config.num_heads = 20
config.grad_clip_value = 100
config.grad_clip_value_gd = 100
config.lr = 0.001
config.wd = 0.0
config.init_scale = 0.002 / config.num_layers
config.bs = 2048
config.bs_gd_train = 2048
config.gd_lr = 0.0003

config.dropout_rate = 0.0
data_creator = vmap(create_circ_cls_data,
                    in_axes=(0, None, None, None, None, None),
                    out_axes=0)

config.y_update = False
config.input_range = 2.0
config.seed = 0

config.analyse = False
config.input_mlp = True
config.input_mlp_out_dim = 120
config.widening_factor = 4
config.sum_norm = False


config.in_proj = True
config.emb_size = 120
config.num_seeds = num_seeds

change_dataloader()

def concate_feature_label_collate(batch):
  features, labels = zip(*batch)
  # append labels to the features
  concatenated = [np.concatenate([f, np.array([l])]) for f, l in zip(features, labels)]
  return np.array(concatenated, dtype=jnp.float32)

def numpy_collate(batch):
  return tree_map(np.asarray, concate_feature_label_collate(batch))

class NumpyLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)

class FlattenAndCast(object):
  def __call__(self, pic):
    return np.ravel(np.array(pic, dtype=jnp.float32))

def prepare_mnist_dataset(seed=0):
    # Set the seed
    np.random.seed(seed)
    # Define our dataset, using torch datasets
    mnist_dataset = MNIST('./data', train=True, download=True, transform=FlattenAndCast())
    training_generator = NumpyLoader(mnist_dataset, batch_size=1, num_workers=0)
    # convert to a np array
    mnist_data = np.array(list(training_generator))

    # squeeze the 2nd dimension
    mnist_data = np.squeeze(mnist_data, axis=1)

    assert total_classes >= num_classes
    filtered_mnist_data = mnist_data[mnist_data[..., -1] < num_classes]

    # batch the data
    num = filtered_mnist_data.shape[0] // bs * bs
    batched_mnist_data = np.asarray(np.split(filtered_mnist_data[:num], bs))
    batched_mnist_data = batched_mnist_data.transpose(1, 0, 2)

    augmented_batched_mnist_data = []
    for i, batch in enumerate(batched_mnist_data):
        augmented_batch = np.copy(batch)
        for _ in range(4):
            # random permutation of total_classes
            perm = np.random.permutation(total_classes)
            augmented_batch[..., -1] = perm[augmented_batch[..., -1].astype(int)]
            augmented_batched_mnist_data.append(np.copy(augmented_batch))

    augmented_batched_mnist_data = jnp.array(augmented_batched_mnist_data)
    # randomly shuffle the data with numpy
    augmented_batched_mnist_data = augmented_batched_mnist_data[np.random.permutation(augmented_batched_mnist_data.shape[0])]
    batched_mnist_data = augmented_batched_mnist_data

    split = np.split(batched_mnist_data, [int(0.8 * batched_mnist_data.shape[0])])
    train_set = split[0]
    test_set = split[1]

    train_set = jnp.array(train_set)
    train_mnist_data = train_set.at[:, -1, -1].set(-1) # -1 for null label
    train_mnist_y = train_set[:, -1]

    test_set = jnp.array(test_set)
    test_mnist_data = test_set.at[:, -1, -1].set(-1) # -1 for null label
    test_mnist_y = test_set[:, -1]

    # pack into train_data and eval_data
    train_data = (train_mnist_data, train_mnist_y)
    eval_data = (test_mnist_data, test_mnist_y)

    return train_data, eval_data


def pretrain(train_data, use_pretrained_weight=False, pretrined_weight_path=None):
    #@title Training
    for cur_seed in range(config.num_seeds):
        if cur_seed == 1:
            save_train_params = train_state.params
        config.seed = cur_seed  
        optimiser, train_state, _, rng = init(train_data)
        if use_pretrained_weight:
            if not pretrined_weight_path:
                raise ValueError("pretrined_weight_path is required")
            from utils import load_weights
            res = load_weights(train_state.params, pretrined_weight_path)
            print(f"loaded pretrained weights from {pretrined_weight_path}")
            return res
        for step in range(config.training_steps):
            train_state, metrics = update(train_state, train_data, optimiser)

    if config.num_seeds == 1:
        save_train_params = train_state.params
    
    return save_train_params
    

def run(perturb_num, alpha, save_train_params, num_trials=10):
    acc_orig_trans = []
    acc_rem_low_trans = []
    acc_rem_high_trans = []
    acc_rem_random_trans = []

    for seed in range(num_trials):

        print(f"running seed {seed}")

        cur_acc_orig_trans = []
        cur_acc_rem_low_trans = []
        cur_acc_rem_high_trans = []
        cur_acc_rem_random_trans = []

        eval_rng = jax.random.PRNGKey(seed)
        def get_embedding(data):
            return predict_stack.apply(save_train_params, eval_rng, data, False)

        _, eval_data = prepare_mnist_dataset(seed)

        for tr_idx in range(eval_data[0].shape[0]):

            if (tr_idx + 1) % 100 == 0:
                print(f"processing {tr_idx + 1}th data")

            X = eval_data[0]
            Y = eval_data[1]

            X_only = X.at[:, :, -1].set(-1)

            emb = get_embedding(X_only)[tr_idx]
            gt = X[tr_idx][:, -1]

            X_b = emb[:-1]
            n_features = X_b.shape[1]
            y = gt[:-1]
            ont_hot_y = jax.nn.one_hot(y, config.output_size)

            w = np.linalg.inv(X_b.T.dot(X_b) + alpha * np.eye(n_features)).dot(X_b.T).dot(ont_hot_y)

            def calc_influence(idx):
                # the first 10 are inputs, the last one is test
                emb_train_all = emb[:-1]
                emb_train = emb[idx:idx+1]
                emb_test = emb[-1:]

                gt_train = X[tr_idx][idx:idx+1, -1]
                gt_test = Y[tr_idx][-1]

                k = emb_train_all.T @ emb_train_all
                # influence of the training data on w
                pred = emb_train @ w
                gt_train_one_hot = jax.nn.one_hot(gt_train, config.output_size)
                infl_w = np.linalg.solve(k + alpha * jnp.eye(k.shape[0]), emb_train.T @ (pred - gt_train_one_hot) + alpha * w)
                # influence of a test point on w
                gt_test_one_hot = jax.nn.one_hot(gt_test, config.output_size)
                d_loss = emb_test.T @ (emb_test @ w - gt_test_one_hot) + alpha * w
                return (jnp.ravel(d_loss) @ jnp.ravel(infl_w)).item()

            gt = Y[tr_idx][-1]

            # compute the prediction error of the test point
            pred_trans = predict.apply(save_train_params, eval_rng, X[tr_idx:tr_idx+1], False)
            pred_trans = jnp.argmax(pred_trans[:, -1])
            acc = 1 if pred_trans == int(gt) else 0
            cur_acc_orig_trans.append(acc)

            infls = []
            for i in range(config.dataset_size):
                infl = calc_influence(i)
                infls.append(infl)
            infls = jnp.array(infls)

            # remvoe the points with the lowest influence
            largest_idx = infls.argsort()[perturb_num:]
            # add the test point index
            largest_idx = jnp.append(largest_idx, config.dataset_size)
            largest_idx.sort()
            X_cleaned = X[:, largest_idx]

            pred_trans = predict.apply(save_train_params, eval_rng, X_cleaned[tr_idx:tr_idx+1], False)
            pred_trans = jnp.argmax(pred_trans[:, -1])
            acc = 1 if pred_trans == int(gt) else 0
            cur_acc_rem_low_trans.append(acc)

            # remvoe the points with the highest influence
            smallest_idx = infls.argsort()[:-perturb_num]
            # add the test point index
            smallest_idx = jnp.append(smallest_idx, config.dataset_size)
            smallest_idx.sort()
            X_cleaned = X[:, smallest_idx]

            pred_trans = predict.apply(save_train_params, eval_rng, X_cleaned[tr_idx:tr_idx+1], False)
            pred_trans = jnp.argmax(pred_trans[:, -1])
            acc = 1 if pred_trans == int(gt) else 0
            cur_acc_rem_high_trans.append(acc)

            # remove random points
            random_idx = jax.random.choice(eval_rng, config.dataset_size, (config.dataset_size - perturb_num, ), replace=False)
            random_idx = jnp.append(random_idx, config.dataset_size)
            random_idx.sort()
            X_cleaned = X[:, random_idx]

            pred_trans = predict.apply(save_train_params, eval_rng, X_cleaned[tr_idx:tr_idx+1], False)
            pred_trans = jnp.argmax(pred_trans[:, -1])
            acc = 1 if pred_trans == int(gt) else 0
            cur_acc_rem_random_trans.append(acc)
        
        acc_orig_trans.append(np.mean(cur_acc_orig_trans))
        acc_rem_low_trans.append(np.mean(cur_acc_rem_low_trans))
        acc_rem_high_trans.append(np.mean(cur_acc_rem_high_trans))
        acc_rem_random_trans.append(np.mean(cur_acc_rem_random_trans))
    
    # dump the results in npy
    np.save(f"{save_dir}/acc_orig_trans_{perturb_num}_{alpha}.npy", np.array(acc_orig_trans))
    np.save(f"{save_dir}/acc_rem_low_trans_{perturb_num}_{alpha}.npy", np.array(acc_rem_low_trans))
    np.save(f"{save_dir}/acc_rem_high_trans_{perturb_num}_{alpha}.npy", np.array(acc_rem_high_trans))
    np.save(f"{save_dir}/acc_rem_random_trans_{perturb_num}_{alpha}.npy", np.array(acc_rem_random_trans))
    print("saved the results")



if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--perturb_num", type=int, nargs="+", default=[1,2,3,4,5,6])
    args.add_argument("--use_pretrained", action="store_true")
    args.add_argument("--alpha", type=float, default=1.0)

    args = args.parse_args()

    use_pretrained = args.use_pretrained
    save_path = "data/mnist_pretrained.pkl"

    train_data, _ = prepare_mnist_dataset()
    save_train_params = pretrain(train_data, use_pretrained_weight=use_pretrained, pretrined_weight_path=save_path)

    save_weights(save_train_params, save_path)

    perturb_nums = args.perturb_num
    alpha = args.alpha

    for perturb_num in perturb_nums:
        run(perturb_num, alpha, save_train_params, num_trials=10)
