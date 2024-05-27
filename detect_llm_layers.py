import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
import os

from utils import (
    load_model,
    process_raw_data,
    get_context,
    perturb_labels,
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


def get_all_label_ids(labels, model_name):
    label_ids = []
    for label in labels:
        if model_name == "vicuna-7b" or model_name == "Llama-2-7b" or model_name == "mistral-7b":
            label_id = tokenizer.convert_tokens_to_ids(f"▁{label}")
        elif model_name == "gpt2-xl" or model_name == "falcon-7b":
            label_id = tokenizer.convert_tokens_to_ids(f"Ġ{label}")
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




def run(num_remove, layer_num):
    global map_int_to_abcd, map_abcd_to_int
    all_fraction_checked_infl = np.zeros((num_trials, icl_dataset_size))

    for seed in range(num_trials):
        data_input, data_output, map_abcd_to_int, map_int_to_abcd = process_raw_data(dataset_name, seed, bs * (icl_dataset_size + 1))
        dataset = get_data(data_input, data_output, bs=bs, num_data=icl_dataset_size)

        fraction_checked_infl = [0 for _ in range(icl_dataset_size)]

        for demo_data, eval_data in dataset:
            # perturb the first {num_remove} labels of the demo data
            remove_indices = np.random.choice(len(demo_data[0]), num_remove, replace=False)
            demo_data_perturb = perturb_labels(demo_data, remove_indices, map_abcd_to_int, map_int_to_abcd)

            infls = compute_infl(
                model_name,
                model,
                tokenizer,
                map_abcd_to_int,
                demo_data_perturb, 
                eval_data, 
                [layer_num], 
                project_dim=project_dim, 
                device=device, 
                score="self", 
                alpha=1e-9
            )
            
            # check infl
            check_idx = np.argsort(infls)[::-1]
            identified = 0
            for pos, idx in enumerate(check_idx):
                if idx in remove_indices:
                    identified += 1
                fraction_checked_infl[pos] += identified

        fraction_checked_infl = np.asarray(fraction_checked_infl) / (len(dataset) * num_remove)

        print(f"fraction_checked_infl: {fraction_checked_infl.mean()};")

        all_fraction_checked_infl[seed] = fraction_checked_infl
    
    return all_fraction_checked_infl


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--layer_nums", type=int, nargs="+", default=[15])
    args.add_argument("--model", type=str, default="vicuna-7b")
    args.add_argument("--dataset_name", type=str, default="ag_news")
    args.add_argument("--project_dim", type=int, default=None)
    args.add_argument("--bs", type=int, default=100)
    args = args.parse_args()

    device = "cuda"
    model_name = args.model
    icl_dataset_size = 20
    project_dim = args.project_dim
    bs = args.bs
    num_trials = 10
    dataset_name = args.dataset_name
    save_dir = f"results/detect_llm_layers_{dataset_name}/"
    os.makedirs(save_dir, exist_ok=True)
    layer_nums = args.layer_nums

    if model_name == "vicuna-7b":
        model_path = "lmsys/vicuna-7b-v1.3"
    elif model_name == "Llama-2-7b":
        model_path = "meta-llama/Llama-2-7b-chat-hf"
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

    for layer_num in args.layer_nums:
        num_remove = 4
        all_fraction_checked_infl = run(num_remove, layer_num)
        print(f"Finishied running for removing {num_remove} data points")
        # save the results
        np.save(os.path.join(save_dir, f"fraction_checked_infl_{model_name}_{num_remove}_{layer_num}.npy"), all_fraction_checked_infl)
