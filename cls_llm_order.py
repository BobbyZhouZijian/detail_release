import itertools
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import os
import json

from utils import DemosTemplate, EvalTemplate
from datasets import load_dataset
from src.infl_torch import calc_influence, get_ridge_weights

from utils import (
    load_model,
    process_raw_data,
    perturb_labels,
    check_answers,
)

import argparse


def get_data(data_input, data_output, map_dict, seed=0, bs=200, num_data=10):

    # first split data_input and data_output into train, val, and test
    assert 6 * num_data <= len(data_input) - bs * num_data, "Not enough data for test"
    test_len = 6 * num_data
    test_input = data_input[-test_len:]
    test_output = data_output[-test_len:]

    dataset = []
    for i in range(bs):
        cur_input = data_input[i*num_data:(i+1)*num_data]
        cur_output = data_output[i*num_data:(i+1)*num_data]
        cur_demo_data = (cur_input, cur_output)
        cur_eval_data = (test_input, test_output)
        dataset.append((cur_demo_data, cur_eval_data))
    return dataset



def run(num_perms):
    # for seed in range(num_trials):
    seed = 0
    num_perms = 2**(int(np.log2(num_perms)))
    data_input, data_output, map_abcd_to_int, map_int_to_abcd = process_raw_data(dataset_name, seed, bs * (icl_dataset_size + 10))
    dataset = get_data(data_input, data_output, map_int_to_abcd, seed=seed, bs=bs, num_data=icl_dataset_size)

    all_infs = []

    for idx, (demo_data, eval_data) in enumerate(dataset):

        max_infl_idx = len(demo_data[0]) - 1
        demo_data = perturb_labels(demo_data, [max_infl_idx], map_abcd_to_int, map_int_to_abcd)

        eval_len = 0
        test_len = 8
        batch_size = 4
        eval_data, test_data = (eval_data[0][:eval_len], eval_data[1][:eval_len]), (eval_data[0][eval_len:eval_len+test_len], eval_data[1][eval_len:eval_len+test_len])

        # cut into 5 intervals
        # intervals = [0,2,4,6,8,10]
        intervals = np.arange(0, len(demo_data[0]))
        infl_intervals = [[] for _ in range(len(intervals))]

        from utils import SobolPermutations
        perms = SobolPermutations(num_perms, icl_dataset_size - 1)


        # consider all permutations of the labels
        for _ in range(num_perms):
            perm = perms[_]

            for i, split_idx in enumerate(intervals):
                cur_perm = perm[perm != max_infl_idx]
                cur_perm = np.concatenate([cur_perm[:split_idx], [max_infl_idx], cur_perm[split_idx:]])
                new_demo_data = ([demo_data[0][p] for p in cur_perm], [demo_data[1][p] for p in cur_perm])
                orig_acc = 0
                for idx in range(test_len // batch_size):
                    cur_test_data = (test_data[0][idx*batch_size:(idx+1)*batch_size], test_data[1][idx*batch_size:(idx+1)*batch_size])
                    orig_acc += check_answers(model, tokenizer, new_demo_data, cur_test_data, device=device)
                orig_acc /= test_len
                infl_intervals[i].append(orig_acc)
        
        infl_res = np.mean(infl_intervals, axis=1)
        all_infs.append(infl_res)

        print(f"Finished running for one demo data\ninfl at all intervals: {infl_res}")
    
    return np.asarray(all_infs)


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--layer_nums", type=int, nargs="+", default=[15])
    args.add_argument("--model", type=str, default="vicuna-7b")
    args.add_argument("--project_dim", type=int, default=None)
    args.add_argument("--num_trials", type=int, default=100)
    args.add_argument("--dataset_name", type=str, default="ag_news")
    args = args.parse_args()

    device = "cuda"
    model_name = args.model
    icl_dataset_size = 20
    bs = args.num_trials
    project_dim = args.project_dim
    dataset_name = args.dataset_name
    save_dir = f"results/cls_llm_order_{dataset_name}/"
    os.makedirs(save_dir, exist_ok=True)
    layer_nums = args.layer_nums

    if model_name == "vicuna-7b":
        model_path = "lmsys/vicuna-7b-v1.3"
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")

    model, tokenizer = load_model(model_path=model_path, device=device)
    # raw_dataset = load_dataset(dataset_name)

    for num_perms in [8]:
        infls = run(num_perms)
        print(f"Finishied running for removing {num_perms} data points")
        # save the results
        np.save(os.path.join(save_dir, f"all_infl_{num_perms}_{model_name}.npy"), infls)
