import torch
import numpy as np
import os
import time
from src.infl_torch import calc_influence_single

from utils import (
    load_model,
    process_raw_data,
    perturb_labels,
    remove_data,
    get_context,
    get_label_pos,
    get_embedding,
    compute_infls,
)

import inseq

import argparse

from gpt_query import model_from_config
import yaml


def load_inseq(model_path, method_name):
    model_kwargs = {
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
    }
    tokenizer_kwargs = {
        "padding_side": "left",
    }
    return inseq.load_model(model_path, method_name, model_kwargs=model_kwargs, tokenizer_kwargs=tokenizer_kwargs)


def inseq_attr_score(inputs, eval_data, model):
    label = eval_data[1][0]
    generated_output = inputs + " " + label
    out = model.attribute(
        input_texts=inputs,
        generation_args={"max_new_tokens": 3},
        n_steps=100,
        internal_batch_size=1,
        generated_texts=generated_output,
        show_progress=False,
    )

    assert model_name == "vicuna-7b" or model_name == "Llama-2-7b" or model_name == "mistral-7b" or model_name == "wizardlm-7b"

    prefix = "▁"

    # find the token id for A and B
    a_token = f"{prefix}A"
    b_token = f"{prefix}B"
    c_token = f"{prefix}C"
    d_token = f"{prefix}D"
    e_token = f"{prefix}E"
    output_token = f"Output"

    all_tokens = [a_token, b_token, c_token, d_token, e_token]

    label_pos = []

    token_w_ids = out.sequence_attributions[0].target

    for pos, token_w_id in enumerate(token_w_ids[:-1]):
        if pos < 2:
            continue
        token = token_w_id.token
        pre_token = token_w_ids[pos-2].token
        if token in all_tokens and pre_token == output_token:
            label_pos.append(pos)
    
    token_attrs = out.sequence_attributions[0].aggregate().target_attributions
    sample_attrs = []
    start_pos = 1  # skip <s>

    for pos in label_pos:
        sample_attrs.append(token_attrs[start_pos:pos+1].sum().item())
        start_pos = pos + 1
    
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


def get_demos(demo_data):
    demos = []
    for i in range(len(demo_data[0])):
        demos.append(f"Input: {demo_data[0][i]}\nOutput: {demo_data[1][i]}")
    return demos


def check_answers_integrated_model(model, demo_data, eval_data):
    contexts = [get_context(demo_data, (eval_data[0][idx:idx+1], eval_data[1][idx:idx+1])) for idx in range(len(eval_data[0]))]
    answers = model.generate_text(contexts, 1, 0, use_seed=True)
    # check if the answer is corret
    num_correct = 0
    for idx in range(len(eval_data[0])):
        if answers[idx].strip() == eval_data[1][idx].strip():
            num_correct += 1
    return num_correct


def remove_and_get_correct_num(demo_data, test_data, remove_idx, model, tokenizer=None, batch_size=4, is_corrupt=False, device="cuda"):
    if is_corrupt:
        new_demo_data = perturb_labels(demo_data, remove_idx, map_abcd_to_int, map_int_to_abcd)
    else:
        new_demo_data = remove_data(demo_data, remove_idx)

    num_correct = 0
    for idx in range(len(test_data[0]) // batch_size):
        cur_test_data = (test_data[0][idx*batch_size:(idx+1)*batch_size], test_data[1][idx*batch_size:(idx+1)*batch_size])
        num_correct += check_answers_integrated_model(model, new_demo_data, cur_test_data)
        # if tokenizer is None:
        #     num_correct += check_answers_integrated_model(model, new_demo_data, cur_test_data)
        # else:
        #     num_correct += check_answers(model, tokenizer, new_demo_data, cur_test_data, device=device)
    return num_correct

def update_config(config, base_config=None):
    # Get default config from yaml
    with open(os.path.join(os.path.dirname(__file__), base_config)) as f:
        default_config = yaml.safe_load(f)

    # Update default config with user config
    # Note that the config is a nested dictionary, so we need to update it recursively
    def update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    return update(default_config, config)

def compute_nguyen_infl(total_iter, num_shot, demo_data, eval_data, model, tokenizer, batch_size):
    icl_dataset_size = len(demo_data[0])
    sampled_indices = torch.ones(total_iter, icl_dataset_size).multinomial(num_shot, replacement=False).to(device)

    with torch.no_grad():
        all_reward = []
        for subset_indices in sampled_indices:
            all_indices = torch.arange(icl_dataset_size).to(device)
            new_demo_data = ([demo_data[0][i] for i in all_indices[subset_indices]], [demo_data[1][i] for i in all_indices[subset_indices]])
            score = remove_and_get_correct_num(new_demo_data, eval_data, [], model, tokenizer, batch_size=batch_size, device=device)
            all_reward.append(score)
    all_reward = np.array(all_reward)
    all_influence = []
    for data_idx in range(icl_dataset_size):
        contain_set_idx = []
        not_contain_set_idx = []
        for subset_idx in range(total_iter):
            if data_idx in sampled_indices[subset_idx]:
                contain_set_idx.append(subset_idx)
            else:
                not_contain_set_idx.append(subset_idx)
        if len(contain_set_idx) == 0 or len(not_contain_set_idx) == 0:
            influence = 0
        else:
            influence = all_reward[contain_set_idx].mean() - all_reward[not_contain_set_idx].mean()
        all_influence.append(influence)
    return np.array(all_influence)


def compute_vinay_infl(demo_data, eval_data, model, tokenizer):
    demos = get_demos(demo_data)
    embeddings = []
    with torch.no_grad():
        for demo in demos:
            input_ids = tokenizer.encode(demo, return_tensors="pt").to(device)
            out = model(input_ids, output_hidden_states=True)
            hidden_states = out.hidden_states
            label_pos = get_label_pos(tokenizer, input_ids[0], model_name=model_name)[:-1]
            emb = get_embedding(hidden_states, label_pos, -1, device=device).detach().float()
            embeddings.append(emb)
    
    num_classes = len(map_abcd_to_int)
    
    linear_model = torch.nn.Linear(embeddings[0].shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(linear_model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(20):
        optimizer.zero_grad()
        for idx in range(len(embeddings)):
            emb = embeddings[idx]
            y = torch.tensor([map_abcd_to_int[demo_data[1][idx]]]).to(device)
            output = linear_model(emb)
            loss = criterion(output, y)
            loss.backward()
        optimizer.step()
    
    eval_demos = get_demos(eval_data)
    eval_embeddings = []

    with torch.no_grad():
        for demo in eval_demos:
            input_ids = tokenizer.encode(demo, return_tensors="pt").to(device)
            out = model(input_ids, output_hidden_states=True)
            hidden_states = out.hidden_states
            label_pos = get_label_pos(tokenizer, input_ids[0], model_name=model_name)[:-1]
            emb = get_embedding(hidden_states, label_pos, -1, device=device).detach().float()
            eval_embeddings.append(emb)
        
    train_labels = torch.tensor([map_abcd_to_int[label] for label in demo_data[1]]).to(device)
    eval_labels = torch.tensor([map_abcd_to_int[label] for label in eval_data[1]]).to(device)

    train_embeddings = torch.cat(embeddings)
    eval_embeddings = torch.cat(eval_embeddings)

    train_dataset = torch.utils.data.TensorDataset(train_embeddings, train_labels)
    eval_dataset = torch.utils.data.TensorDataset(eval_embeddings, eval_labels)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    eval_data_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False)

    infls = [0. for _ in range(len(demo_data[0]))]

    for idx in range(len(eval_data[0])):
        cur_infls = calc_influence_single(linear_model, train_data_loader, eval_data_loader, idx, 0, len(demo_data[0]) // 2, 2)
        for i in range(len(demo_data[0])):
            infls[i] += cur_infls[i].item()

    return np.array(infls)


def run(num_remove, gpt_model):
    global model, tokenizer, map_abcd_to_int, map_int_to_abcd
    acc_rem_low = []
    wall_time = []

    if method_name == "ig":
        model = load_inseq(model_path, "integrated_gradients")
    elif method_name == "sig":
        model = load_inseq(model_path, "sequential_integrated_gradients")
    elif method_name == "reagent":
        model = load_inseq(model_path, "reagent")
    elif method_name == "attention":
        model = load_inseq(model_path, "attention")
    elif method_name == "lime":
        model = load_inseq(model_path, "lime")
    elif method_name in ["influence", "nguyen_infl", "vinay_infl", "random"]:
        model, tokenizer = load_model(model_path=model_path, device=device)
    else:
        raise NotImplementedError(f"Method {method_name} not implemented")

    for seed in range(num_trials):
        data_input, data_output, map_abcd_to_int, map_int_to_abcd = process_raw_data(dataset_name, seed, bs * (icl_dataset_size) + 1000)

        eval_len = 20
        test_len = 20
        batch_size = 1

        dataset = get_data(data_input, data_output, bs=bs, num_data=icl_dataset_size, eval_len=eval_len, test_len=test_len)

        assert eval_len % batch_size == 0 and test_len % batch_size == 0, "Eval and test length must be divisible by batch size"


        for demo_data, eval_data in dataset:

            # randomly perturb 5 labels
            perturb_indices = np.random.choice(len(demo_data[0]), 5, replace=False)
            demo_data = perturb_labels(demo_data, perturb_indices, map_abcd_to_int, map_int_to_abcd)

            num_success_rem_low = 0

            eval_data, test_data = (eval_data[0][:eval_len], eval_data[1][:eval_len]), (eval_data[0][eval_len:eval_len+test_len], eval_data[1][eval_len:eval_len+test_len])
            start, end = None, None

            if method_name == "influence":
                # infl
                start = time.time()
                infls = np.zeros(len(demo_data[0]))
                for idx in range(eval_len):
                    cur_eval_data = (eval_data[0][idx:idx+1], eval_data[1][idx:idx+1])
                    context = get_context(demo_data, cur_eval_data)
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
                        alpha=alpha,
                    )
                end = time.time()
                # remove the top {num_remove} data points with the lowest influence
                remove_idx = np.argsort(infls)[:num_remove]
                num_success_rem_low += remove_and_get_correct_num(demo_data, test_data, remove_idx, gpt_model, tokenizer, batch_size=batch_size, is_corrupt=corrupt, device=device)

            elif method_name in ["ig", "sig", "attention", "reagent", "lime"]:    
                start = time.time()       
                attrs = np.zeros(len(demo_data[0]))
                for idx in range(eval_len):
                    cur_eval_data = (eval_data[0][idx:idx+1], eval_data[1][idx:idx+1])
                    context = get_context(demo_data, cur_eval_data)
                    attrs += inseq_attr_score(context, cur_eval_data, model)
                end = time.time()
                remove_idx = np.argsort(attrs)[:num_remove]
                num_success_rem_low += remove_and_get_correct_num(demo_data, test_data, remove_idx, gpt_model, tokenizer, batch_size=batch_size, is_corrupt=corrupt, device=device)
            elif method_name == "nguyen_infl":
                start = time.time()
                attrs = compute_nguyen_infl(100, 1, demo_data, eval_data, model, tokenizer, batch_size)
                end = time.time()
                remove_idx = np.argsort(attrs)[:num_remove]
                num_success_rem_low += remove_and_get_correct_num(demo_data, test_data, remove_idx, gpt_model, tokenizer, batch_size=batch_size, is_corrupt=corrupt, device=device)
            elif method_name == "vinay_infl":
                start = time.time()
                attrs = compute_vinay_infl(demo_data, eval_data, model, tokenizer)
                end = time.time()
                remove_idx = np.argsort(attrs)[:num_remove]
                num_success_rem_low += remove_and_get_correct_num(demo_data, test_data, remove_idx, gpt_model, tokenizer, batch_size=batch_size, is_corrupt=corrupt, device=device)
            elif method_name == "random":
                start = time.time()
                remove_idx = np.random.choice(len(demo_data[0]), num_remove, replace=False)
                end = time.time()
                num_success_rem_low += remove_and_get_correct_num(demo_data, test_data, remove_idx, gpt_model, tokenizer, batch_size=batch_size, is_corrupt=corrupt, device=device)
            else:
                raise NotImplementedError(f"Method {method_name} not implemented")
            
            print(f"Success Remove low: {num_success_rem_low};")

            cur_acc_rem_low = num_success_rem_low / (test_len)
            acc_rem_low.append(cur_acc_rem_low)
            wall_time.append(end - start)

            print(f"Acc Remove low: {cur_acc_rem_low}; wall time: {end - start}")
    
    return acc_rem_low, wall_time


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--layer_nums", type=int, nargs="+", default=[15])
    args.add_argument("--model", type=str, default="vicuna-7b")
    args.add_argument("--dataset_name", type=str, default="ag_news")
    args.add_argument("--project_dim", type=int, default=None)
    args.add_argument("--use_label_pos", action="store_true")
    args.add_argument("--num_trials", type=int, default=100)
    args.add_argument("--method_name", type=str, default="influence")
    args.add_argument("--num_removes", type=int, nargs="+", default=[10])
    args.add_argument("--alpha", type=float, default=1.0)
    args.add_argument("--corrupt", action="store_true")
    args.add_argument("--gpt", type=str, default='gpt-3.5-turbo-1106')

    args = args.parse_args()

    device = "cuda"
    model_name = args.model
    icl_dataset_size = 20
    project_dim = args.project_dim
    num_removes = args.num_removes
    alpha = args.alpha
    bs = args.num_trials
    num_trials = 1  # a quick hack; to be removed
    method_name = args.method_name
    dataset_name = args.dataset_name
    save_dir = f"results/cls_llm_other_{dataset_name}/"
    os.makedirs(save_dir, exist_ok=True)
    layer_nums = args.layer_nums
    corrupt = args.corrupt
    use_label_pos = args.use_label_pos

    assert method_name in ["influence", "ig", "sig", "attention", "lime", "reagent", "nguyen_infl", "vinay_infl", "random"], f"Method {method_name} not implemented"

    if model_name == "vicuna-7b":
        model_path = "lmsys/vicuna-7b-v1.3"
    elif model_name == "Llama-2-7b":
        model_path = "meta-llama/Llama-2-7b-chat-hf"
    elif model_name == "Llama-2-13b":
        model_path = "meta-llama/Llama-2-13b-chat-hf"
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

    model, tokenizer = None, None
    # raw_dataset = load_dataset(dataset_name)
    gpt_config = update_config({'evaluation': {'model': {'gpt_config': {'model': args.gpt}}}}, 'gpt_config.yaml')
    gpt_model = model_from_config(gpt_config['evaluation']['model'])
    
    for num_remove in num_removes:
        acc_rem_low, wall_time = run(num_remove, gpt_model)
        print(f"Finishied running for removing {num_remove} data points")
        # save the results
        np.save(os.path.join(save_dir, f"acc_rem_low_{method_name}_{model_name}_{num_remove}_{'remove' if not corrupt else 'corrupt'}{'_use_label_pos' if use_label_pos else ''}.npy"), acc_rem_low)
        np.save(os.path.join(save_dir, f"wall_time_{method_name}_{model_name}_{num_remove}_{'remove' if not corrupt else 'corrupt'}{'_use_label_pos' if use_label_pos else ''}.npy"), wall_time)
