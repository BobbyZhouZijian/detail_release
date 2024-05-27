import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
import os
import time

from utils import (
    load_model,
    process_raw_data,
    perturb_labels,
    compute_infl,
    get_context,
    remove_data,
)

import argparse


def get_data(data_input, data_output, bs=200, num_data=10):
    dataset = []
    for i in range(bs):
        cur_input = data_input[i*num_data:(i+1)*num_data]
        cur_output = data_output[i*num_data:(i+1)*num_data]
        cur_eval_input = data_input[i*num_data+num_data:i*num_data+num_data+1]
        cur_eval_output = data_output[i*num_data+num_data:i*num_data+num_data+1]
        cur_demo_data = (cur_input, cur_output)
        cur_eval_data = (cur_eval_input, cur_eval_output)
        dataset.append((cur_demo_data, cur_eval_data))
    return dataset


def get_all_label_ids(labels, model_name):
    label_ids = []
    for label in labels:
        if model_name in ["vicuna-7b", "Llama-2-7b", "Llama-2-13b", "mistral-7b"]:
            label_id = tokenizer.convert_tokens_to_ids(f"▁{label}")
        elif model_name in ["gpt2-xl", "falcon-7b", "mamba"]:
            label_id = tokenizer.convert_tokens_to_ids(f"Ġ{label}")
        else:
            raise NotImplementedError(f"Model {model_name} not implemented")
        label_ids.append(label_id)
    return label_ids


def compute_loo(demo_data, eval_data, model, tokenizer, model_name="vicuna-7b", device="cuda"):

    def get_loss(input_ids):
        res = model.generate(input_ids, do_sample=False, max_new_tokens=3, pad_token_id=tokenizer.eos_token_id, output_scores=True, return_dict_in_generate=True)
        labels = list(map_abcd_to_int.keys())
        label_ids = get_all_label_ids(labels, model_name)
        all_scores = res.scores[0][0][label_ids]
        output_scores = F.softmax(all_scores, dim=0)
        correct_label = eval_data[1][0]
        label = map_abcd_to_int[correct_label]
        label = torch.tensor([label]).to(device)
        label = torch.nn.functional.one_hot(label, num_classes=output_scores.shape[-1]).reshape(-1).float()
        loss = CrossEntropyLoss()(output_scores, label).item()
        return loss

    # compute original loss
    context = get_context(demo_data, eval_data)
    input_ids = tokenizer.encode(context, return_tensors="pt").to(device)
    orig_loss = get_loss(input_ids)

    loo_scores = [0 for _ in range(len(demo_data[0]))]

    for i in range(len(demo_data[0])):
        # leave one out
        demo_data_loo = remove_data(demo_data, [i])
        context = get_context(demo_data_loo, eval_data)
        input_ids = tokenizer.encode(context, return_tensors="pt").to(device)
        loo_loss = get_loss(input_ids)
        loo_scores[i] = orig_loss - loo_loss

    return loo_scores


@torch.no_grad()
def run(num_remove):
    global map_int_to_abcd, map_abcd_to_int
    all_fraction_checked_infl = np.zeros((num_trials, icl_dataset_size))
    all_fraction_checked_loo = np.zeros((num_trials, icl_dataset_size))
    all_fraction_checked_random = np.zeros((num_trials, icl_dataset_size))
    all_wall_time_infl = np.zeros((num_trials, bs))
    all_wall_time_loo = np.zeros((num_trials, bs))

    for seed in range(num_trials):
        data_input, data_output, map_abcd_to_int, map_int_to_abcd = process_raw_data(dataset_name, seed, bs * (icl_dataset_size + 1))
        dataset = get_data(data_input, data_output, bs=bs, num_data=icl_dataset_size)

        fraction_checked_infl = [0 for _ in range(icl_dataset_size)]
        fraction_checked_loo = [0 for _ in range(icl_dataset_size)]
        fraction_checked_random = [0 for _ in range(icl_dataset_size)]

        wall_time_infl = []
        wall_time_loo = []

        for demo_data, eval_data in dataset:
            remove_indices = np.random.choice(len(demo_data[0]), num_remove, replace=False)
            demo_data_perturb = perturb_labels(demo_data, remove_indices, map_abcd_to_int, map_int_to_abcd)

            start_infl = time.time()
            infls = compute_infl(
                model_name,
                model,
                tokenizer,
                map_abcd_to_int,
                demo_data_perturb, 
                eval_data, 
                layer_nums, 
                project_dim=project_dim, 
                device=device, 
                score="self",
                alpha=1e-9
            )
            end_infl = time.time()
            wall_time_infl.append(end_infl - start_infl)

            start_loo = time.time()
            loo_check_idx = np.argsort(compute_loo(demo_data_perturb, eval_data, model, tokenizer, model_name=model_name, device=device))[::-1]
            end_loo = time.time()

            wall_time_loo.append(end_loo - start_loo)
            
            # check infl
            check_idx = np.argsort(infls)[::-1]
            identified = 0
            for pos, idx in enumerate(check_idx):
                if idx in remove_indices:
                    identified += 1
                fraction_checked_infl[pos] += identified
            
            # check loo
            identified = 0
            for pos, idx in enumerate(loo_check_idx):
                if idx in remove_indices:
                    identified += 1
                fraction_checked_loo[pos] += identified
        
            # check random
            random_check_idx = np.random.permutation(icl_dataset_size)
            identified = 0
            for pos, idx in enumerate(random_check_idx):
                if idx in remove_indices:
                    identified += 1
                fraction_checked_random[pos] += identified

        fraction_checked_infl = np.asarray(fraction_checked_infl) / (len(dataset) * num_remove)
        fraction_checked_loo = np.asarray(fraction_checked_loo) / (len(dataset) * num_remove)
        fraction_checked_random = np.asarray(fraction_checked_random) / (len(dataset) * num_remove)
        wall_time_infl = np.asarray(wall_time_infl)
        wall_time_loo = np.asarray(wall_time_loo)

        print(f"fraction_checked_infl: {fraction_checked_infl.mean()}; fraction_checked_loo: {fraction_checked_loo.mean()};")
        print(f"wall_time_infl: {wall_time_infl.mean()}; wall_time_loo: {wall_time_loo.mean()}")

        all_fraction_checked_infl[seed] = fraction_checked_infl
        all_fraction_checked_loo[seed] = fraction_checked_loo
        all_fraction_checked_random[seed] = fraction_checked_random
        all_wall_time_infl[seed] = wall_time_infl
        all_wall_time_loo[seed] = wall_time_loo
    
    return all_fraction_checked_infl, all_fraction_checked_loo, all_fraction_checked_random, all_wall_time_infl, all_wall_time_loo


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--layer_nums", type=int, nargs="+", default=[15])
    args.add_argument("--model", type=str, default="vicuna-7b")
    args.add_argument("--dataset_name", type=str, default="ag_news")
    args.add_argument("--project_dim", type=int, default=None)
    args.add_argument("--num_removes", type=int, nargs="+", default=[4])
    args.add_argument("--num_trials", type=int, default=10)
    args.add_argument("--bs", type=int, default=100)
    args = args.parse_args()

    device = "cuda"
    model_name = args.model
    icl_dataset_size = 20
    project_dim = args.project_dim
    bs = args.bs
    num_trials = args.num_trials
    num_removes = args.num_removes
    dataset_name = args.dataset_name
    save_dir = f"results/detect_llm_{dataset_name}/"
    os.makedirs(save_dir, exist_ok=True)
    layer_nums = args.layer_nums

    if model_name == "vicuna-7b":
        model_path = "lmsys/vicuna-7b-v1.3"
    elif model_name == "Llama-2-7b":
        model_path = "meta-llama/Llama-2-7b-chat-hf"
    elif model_name == "Llama-2-13b":
        model_path = "meta-llama/Llama-2-13b-chat-hf"
    elif model_name == "mamba":
        model_path = "state-spaces/mamba-2.8b-hf"
    elif model_name == "gpt2-xl":
        model_path = "gpt2-xl"
    elif model_name == "mistral-7b":
        model_path = "mistralai/Mistral-7B-v0.1"
    elif model_name == "falcon-7b":
        # model_path = "tiiuae/falcon-7b"
        model_path = "OpenBuddy/openbuddy-falcon-7b-v6-bf16"
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")

    model, tokenizer = load_model(model_path=model_path, device=device)
    # raw_dataset = load_dataset(dataset_name)

    for num_remove in num_removes:
        all_fraction_checked_infl, all_fraction_checked_loo, all_fraction_checked_random, all_wall_time_infl, all_wall_time_loo = run(num_remove)
        print(f"Finishied running for removing {num_remove} data points")
        # save the results
        np.save(os.path.join(save_dir, f"fraction_checked_infl_{model_name}_{num_remove}.npy"), all_fraction_checked_infl)
        np.save(os.path.join(save_dir, f"fraction_checked_loo_{model_name}_{num_remove}.npy"), all_fraction_checked_loo)
        np.save(os.path.join(save_dir, f"fraction_checked_random_{model_name}_{num_remove}.npy"), all_fraction_checked_random)
        np.save(os.path.join(save_dir, f"wall_time_infl_{model_name}_{num_remove}.npy"), all_wall_time_infl)
        np.save(os.path.join(save_dir, f"wall_time_loo_{model_name}_{num_remove}.npy"), all_wall_time_loo)
