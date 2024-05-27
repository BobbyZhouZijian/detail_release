import torch
import numpy as np
import os

from utils import (
    process_raw_data, 
    load_model,
    compute_infls,
    check_answers,
    remove_data,
    perturb_labels,
)

import inseq

import argparse


def load_inseq(model_path, method_name):
    return inseq.load_model(model_path, method_name)


def inseq_attr_score(inputs, eval_data, model):
    label = eval_data[1][0]
    generated_output = inputs + " " + label
    out = model.attribute(
        input_texts=inputs,
        generation_args={"max_new_tokens": 3},
        n_steps=100,
        internal_batch_size=1,
        generated_texts=generated_output
    )

    assert model_name == "vicuna-7b" or model_name == "Llama-2-7b" or model_name == "mistral-7b" or model_name == "wizardlm-7b"

    prefix = "▁"

    # find the token id for A and B
    a_token = f"{prefix}A"
    b_token = f"{prefix}B"
    c_token = f"{prefix}C"
    d_token = f"{prefix}D"
    e_token = f"{prefix}E"

    all_tokens = [a_token, b_token, c_token, d_token, e_token]

    label_pos = []

    for pos, token_w_id in enumerate(out.sequence_attributions[0].target[:-1]):
        token = token_w_id.token
        if token in all_tokens:
            label_pos.append(pos - 1)
    
    token_attrs = out.sequence_attributions[0].aggregate().target_attributions
    sample_attrs = []
    start_pos = 1

    for pos in label_pos:
        sample_attrs.append(token_attrs[start_pos:pos+1].sum().item())
        start_pos = pos + 2
    
    return np.array(sample_attrs)


def get_data(data_input, data_output, bs=200, num_data=10, eval_len=20, test_len=20):
    # first split data_input and data_output into train, val, and test
    assert 6 * num_data <= len(data_input) - bs * num_data, "Not enough data for test"
    test_input = data_input[-test_len:]
    test_output = data_output[-test_len:]
    # eval_input = data_input[-(eval_len + test_len):-test_len]
    # eval_output = data_output[-(eval_len + test_len):-test_len]

    dataset = []
    for i in range(bs):
        cur_input = data_input[i*num_data:(i+1)*num_data]
        cur_output = data_output[i*num_data:(i+1)*num_data]
        eval_input = data_input[(i+1)*num_data:(i+1)*num_data+eval_len]
        eval_output = data_output[(i+1)*num_data:(i+1)*num_data+eval_len]
        cur_test_input = eval_input + test_input
        cur_test_output = eval_output + test_output
        cur_demo_data = (cur_input, cur_output)
        cur_eval_data = (cur_test_input, cur_test_output)
        dataset.append((cur_demo_data, cur_eval_data))
    return dataset


@torch.no_grad()
def run(num_remove, start_seed=0):
    acc_orig = []
    acc_rem_high = []
    acc_rem_low = []
    acc_rem_random = []

    for seed in range(num_trials):
        seed += start_seed
        eval_len = 20
        test_len = 120
        batch_size = 1

        data_input, data_output, map_abcd_to_int, map_int_to_abcd = process_raw_data(dataset_name, seed, bs * (icl_dataset_size) + 1000)
        dataset = get_data(data_input, data_output, bs=bs, num_data=icl_dataset_size, eval_len=eval_len, test_len=test_len)

        assert eval_len % batch_size == 0 and test_len % batch_size == 0, "Eval and test length must be divisible by batch size"


        for batch_idx, (demo_data, eval_data) in enumerate(dataset):

            # perturb a new labels
            perturb_indices = np.random.choice(len(demo_data[0]), 5, replace=False)
            demo_data = perturb_labels(demo_data, perturb_indices)

            num_success_orig = 0
            num_success_rem_high = 0
            num_success_rem_low = 0
            num_success_rem_random = 0

            eval_data, test_data = (eval_data[0][:eval_len], eval_data[1][:eval_len]), (eval_data[0][eval_len:eval_len+test_len], eval_data[1][eval_len:eval_len+test_len])

            infls = np.zeros(len(demo_data[0]))
            for idx in range(eval_len // batch_size):
                cur_eval_data = (eval_data[0][idx*batch_size:(idx+1)*batch_size], eval_data[1][idx*batch_size:(idx+1)*batch_size])
                infls += compute_infls(
                    model_name,
                    model,
                    tokenizer,
                    map_abcd_to_int,
                    demo_data, 
                    cur_eval_data, 
                    layer_nums, 
                    project_dim=project_dim, 
                    device=device, 
                    score="test",
                    alpha=alpha
                )
            
            # # check the original context
            for idx in range(test_len // batch_size):
                cur_test_data = (test_data[0][idx*batch_size:(idx+1)*batch_size], test_data[1][idx*batch_size:(idx+1)*batch_size])
                num_success_orig += check_answers(model, tokenizer, demo_data, cur_test_data, device=device)
            
            # remove the top {num_remove} data points with the highest influence
            remove_idx = np.argsort(infls)[-num_remove:]
            if corrupt:
                new_demo_data = perturb_labels(demo_data, remove_idx, map_int_to_abcd, map_abcd_to_int)
            else:
                new_demo_data = remove_data(demo_data, remove_idx)
            
            for idx in range(test_len // batch_size):
                cur_test_data = (test_data[0][idx*batch_size:(idx+1)*batch_size], test_data[1][idx*batch_size:(idx+1)*batch_size])
                num_success_rem_high += check_answers(model, tokenizer, new_demo_data, cur_test_data, device=device)
            
            # remove the top {num_remove} data points with the lowest influence
            remove_idx = np.argsort(infls)[:num_remove]
            if corrupt:
                new_demo_data = perturb_labels(demo_data, remove_idx, map_int_to_abcd, map_abcd_to_int)
            else:
                new_demo_data = remove_data(demo_data, remove_idx)

            for idx in range(test_len // batch_size):
                cur_test_data = (test_data[0][idx*batch_size:(idx+1)*batch_size], test_data[1][idx*batch_size:(idx+1)*batch_size])
                num_success_rem_low += check_answers(model, tokenizer, new_demo_data, cur_test_data, device=device)

            # remove {num_remove} random data points
            remove_idx = np.random.choice(len(demo_data[0]), num_remove, replace=False)
            if corrupt:
                new_demo_data = perturb_labels(demo_data, remove_idx, map_int_to_abcd, map_abcd_to_int)
            else:
                new_demo_data = remove_data(demo_data, remove_idx)
            
            for idx in range(test_len // batch_size):
                cur_test_data = (test_data[0][idx*batch_size:(idx+1)*batch_size], test_data[1][idx*batch_size:(idx+1)*batch_size])
                num_success_rem_random += check_answers(model, tokenizer, new_demo_data, cur_test_data, device=device)
            
            # print(f"Success Original: {num_success_orig}, Success Remove high: {num_success_rem_high}, Success Remove low: {num_success_rem_low}, Success Remove random: {num_success_rem_random}")

            cur_acc_orig = num_success_orig / (test_len)
            cur_acc_rem_high = num_success_rem_high / (test_len)
            cur_acc_rem_low = num_success_rem_low / (test_len)
            cur_acc_rem_random = num_success_rem_random / (test_len)

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
    args.add_argument("--alpha", type=float, default=1.0)
    args.add_argument("--num_trials", type=int, default=100)
    args.add_argument("--corrupt", action="store_true")
    args.add_argument("--append_trials", action="store_true")
    args = args.parse_args()

    device = "cuda"
    model_name = args.model
    icl_dataset_size = 20
    project_dim = args.project_dim
    alpha = args.alpha
    bs = args.num_trials
    num_trials = 1  # a quick hack; to be removed
    dataset_name = args.dataset_name
    save_dir = f"results/cls_llm_comp_{dataset_name}/"
    os.makedirs(save_dir, exist_ok=True)
    layer_nums = args.layer_nums
    corrupt = args.corrupt
    append_trials = args.append_trials
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
        if append_trials:
            old_acc_orig = np.load(os.path.join(save_dir, f"acc_orig_{model_name}_{num_remove}_{'remove' if not corrupt else 'corrupt'}{'_use_label_pos' if use_label_pos else ''}.npy"))
            old_acc_rem_high = np.load(os.path.join(save_dir, f"acc_rem_high_{model_name}_{num_remove}_{'remove' if not corrupt else 'corrupt'}{'_use_label_pos' if use_label_pos else ''}.npy"))
            old_acc_rem_low = np.load(os.path.join(save_dir, f"acc_rem_low_{model_name}_{num_remove}_{'remove' if not corrupt else 'corrupt'}{'_use_label_pos' if use_label_pos else ''}.npy"))
            old_acc_rem_random = np.load(os.path.join(save_dir, f"acc_rem_random_{model_name}_{num_remove}_{'remove' if not corrupt else 'corrupt'}{'_use_label_pos' if use_label_pos else ''}.npy"))
            start_seed = len(old_acc_orig)
            print(f"Appending to existing results for removing {num_remove} data points with start seed: {start_seed}")
        else:
            old_acc_orig = np.array([])
            old_acc_rem_high = np.array([])
            old_acc_rem_low = np.array([])
            old_acc_rem_random = np.array([])
            start_seed = 0
        acc_orig, acc_rem_high, acc_rem_low, acc_rem_random = run(num_remove, start_seed=start_seed)
        print(f"Finishied running for removing {num_remove} data points")
        acc_orig = np.concatenate([old_acc_orig, acc_orig])
        acc_rem_high = np.concatenate([old_acc_rem_high, acc_rem_high])
        acc_rem_low = np.concatenate([old_acc_rem_low, acc_rem_low])
        acc_rem_random = np.concatenate([old_acc_rem_random, acc_rem_random])
        # save the results
        np.save(os.path.join(save_dir, f"acc_orig_{model_name}_{num_remove}_{'remove' if not corrupt else 'corrupt'}{'_use_label_pos' if use_label_pos else ''}.npy"), acc_orig)
        np.save(os.path.join(save_dir, f"acc_rem_high_{model_name}_{num_remove}_{'remove' if not corrupt else 'corrupt'}{'_use_label_pos' if use_label_pos else ''}.npy"), acc_rem_high)
        np.save(os.path.join(save_dir, f"acc_rem_low_{model_name}_{num_remove}_{'remove' if not corrupt else 'corrupt'}{'_use_label_pos' if use_label_pos else ''}.npy"), acc_rem_low)
        np.save(os.path.join(save_dir, f"acc_rem_random_{model_name}_{num_remove}_{'remove' if not corrupt else 'corrupt'}{'_use_label_pos' if use_label_pos else ''}.npy"), acc_rem_random)
