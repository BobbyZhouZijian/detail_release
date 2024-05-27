import torch
import numpy as np
import os

from utils import (
    load_model,
    process_raw_data,
    perturb_labels,
    get_context,
    check_answer,
    remove_data,
    compute_infl,
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


@torch.no_grad()
def run(num_remove):
    acc_orig = []
    acc_rem_high = []
    acc_rem_low = []
    acc_rem_random = []

    for seed in range(num_trials):
        data_input, data_output, map_abcd_to_int, map_int_to_abcd = process_raw_data(dataset_name, seed, bs * (icl_dataset_size + 1))
        dataset = get_data(data_input, data_output, bs=bs, num_data=icl_dataset_size)

        num_success_orig = 0
        num_success_rem_high = 0
        num_success_rem_low = 0
        num_success_rem_random = 0


        for demo_data, eval_data in dataset:
            context = get_context(demo_data, eval_data)

            infls = compute_infl(
                model_name,
                model,
                tokenizer,
                map_abcd_to_int,
                demo_data, 
                eval_data, 
                layer_nums, 
                project_dim=project_dim,
                device=device, 
                score="test",
                alpha=1.0
            )
            
            # check the original context
            if check_answer(model, tokenizer, context, eval_data, device=device):
                num_success_orig += 1
            
            # remove the top {num_remove} data points with the highest influence
            remove_idx = np.argsort(infls)[-num_remove:]
            if corrupt:
                new_demo_data = perturb_labels(demo_data, remove_idx, map_abcd_to_int, map_int_to_abcd)
            else:
                new_demo_data = remove_data(demo_data, remove_idx)
            new_context = get_context(new_demo_data, eval_data)
            if check_answer(model, tokenizer, new_context, eval_data, device=device):
                num_success_rem_high += 1
            
            # remove the top {num_remove} data points with the lowest influence
            remove_idx = np.argsort(infls)[:num_remove]
            if corrupt:
                new_demo_data = perturb_labels(demo_data, remove_idx, map_abcd_to_int, map_int_to_abcd)
            else:
                new_demo_data = remove_data(demo_data, remove_idx)
            new_context = get_context(new_demo_data, eval_data)
            if check_answer(model, tokenizer, new_context, eval_data, device=device):
                num_success_rem_low += 1

            # remove {num_remove} random data points
            remove_idx = np.random.choice(len(demo_data[0]), num_remove, replace=False)
            if corrupt:
                new_demo_data = perturb_labels(demo_data, remove_idx, map_abcd_to_int, map_int_to_abcd)
            else:
                new_demo_data = remove_data(demo_data, remove_idx)
            new_context = get_context(new_demo_data, eval_data)
            if check_answer(model, tokenizer, new_context, eval_data, device=device):
                num_success_rem_random += 1

        cur_acc_orig = num_success_orig / bs
        cur_acc_rem_high = num_success_rem_high / bs
        cur_acc_rem_low = num_success_rem_low / bs
        cur_acc_rem_random = num_success_rem_random / bs

        acc_orig.append(cur_acc_orig)
        acc_rem_high.append(cur_acc_rem_high)
        acc_rem_low.append(cur_acc_rem_low)
        acc_rem_random.append(cur_acc_rem_random)

        print(f"Acc Original: {cur_acc_orig}")
        print(f"Acc Remove high: {cur_acc_rem_high}")
        print(f"Acc Remove low: {cur_acc_rem_low}")
        print(f"Acc Remove random: {cur_acc_rem_random}")
    
    return acc_orig, acc_rem_high, acc_rem_low, acc_rem_random


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--layer_nums", type=int, nargs="+", default=[15])
    args.add_argument("--model", type=str, default="vicuna-7b")
    args.add_argument("--dataset_name", type=str, default="ag_news")
    args.add_argument("--project_dim", type=int, default=None)
    args.add_argument("--use_label_pos", action="store_true")
    args.add_argument("--bs", type=int, default=100)
    args.add_argument("--num_trials", type=int, default=10)
    args.add_argument("--corrupt", action="store_true")
    args = args.parse_args()

    device = "cuda"
    model_name = args.model
    icl_dataset_size = 20
    project_dim = args.project_dim
    bs = args.bs
    num_trials = args.num_trials
    dataset_name = args.dataset_name
    save_dir = f"results/cls_llm_{dataset_name}/"
    os.makedirs(save_dir, exist_ok=True)
    layer_nums = args.layer_nums
    corrupt = args.corrupt
    use_label_pos = args.use_label_pos

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
    elif model_name == "wizardlm-7b":
        model_path = "WizardLM/WizardMath-7B-V1.1"
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")

    model, tokenizer = load_model(model_path=model_path, device=device)
    # raw_dataset = load_dataset(dataset_name)
    
    for num_remove in [4,7,10,13,16]:
        acc_orig, acc_rem_high, acc_rem_low, acc_rem_random = run(num_remove)
        print(f"Finishied running for removing {num_remove} data points")
        # save the results
        np.save(os.path.join(save_dir, f"acc_orig_{model_name}_{num_remove}_{'remove' if not corrupt else 'corrupt'}{'_use_label_pos' if use_label_pos else ''}.npy"), acc_orig)
        np.save(os.path.join(save_dir, f"acc_rem_high_{model_name}_{num_remove}_{'remove' if not corrupt else 'corrupt'}{'_use_label_pos' if use_label_pos else ''}.npy"), acc_rem_high)
        np.save(os.path.join(save_dir, f"acc_rem_low_{model_name}_{num_remove}_{'remove' if not corrupt else 'corrupt'}{'_use_label_pos' if use_label_pos else ''}.npy"), acc_rem_low)
        np.save(os.path.join(save_dir, f"acc_rem_random_{model_name}_{num_remove}_{'remove' if not corrupt else 'corrupt'}{'_use_label_pos' if use_label_pos else ''}.npy"), acc_rem_random)
