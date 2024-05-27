import itertools
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import os
import json

from utils import DemosTemplate, EvalTemplate
from datasets import load_dataset
from src.infl_torch import calc_influence, get_ridge_weights

import argparse

from gpt_query import model_from_config
import yaml


map_int_to_abcd = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
}

map_abcd_to_int = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
}


def load_model(model_path="lmsys/vicuna-7b-v1.3", tokenizer_path=None, device="cuda"):
    if tokenizer_path is None:
        tokenizer_path = model_path
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=False
    )
    return model, tokenizer


def process_raw_data(dataset_name, seed, total_num_data):
    global map_abcd_to_int, map_int_to_abcd
    if dataset_name == "ag_news":
        map_int_to_abcd = {
            0: "A",
            1: "B",
            2: "C",
            3: "D",
        }

        map_abcd_to_int = {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
        }
        raw_dataset = load_dataset(dataset_name)
        data = raw_dataset["train"]
        data = data.shuffle(seed=seed)
        # total_num_data = bs * (num_data + 1) # +1 for test query
        assert len(data) >= total_num_data, f"Dataset size is {len(data)}, which is less than {total_num_data}"
        map_fn = lambda x: map_int_to_abcd[x]
        data_input = data['text'][:total_num_data]
        data_output = list(map(map_fn, data['label'][:total_num_data]))
    elif dataset_name == "rotten_tomatoes":
        map_int_to_abcd = {
            0: "A",
            1: "B",
        }

        map_abcd_to_int = {
            "A": 0,
            "B": 1,
        }
        raw_dataset = load_dataset("rotten_tomatoes")
        data = raw_dataset["train"]
        data = data.shuffle(seed=seed)
        # total_num_data = bs * (num_data + 1) # +1 for test query
        assert len(data) >= total_num_data, f"Dataset size is {len(data)}, which is less than {total_num_data}"
        map_fn = lambda x: map_int_to_abcd[x]
        data_input = data['text'][:total_num_data]
        data_output = list(map(map_fn, data['label'][:total_num_data]))
    elif dataset_name == "sst2":
        map_int_to_abcd = {
            0: "A",
            1: "B",
        }

        map_abcd_to_int = {
            "A": 0,
            "B": 1,
        }
        raw_dataset = load_dataset("SetFit/sst2")
        data = raw_dataset["train"]
        data = data.shuffle(seed=seed)
        # total_num_data = bs * (num_data + 1) # +1 for test query
        assert len(data) >= total_num_data, f"Dataset size is {len(data)}, which is less than {total_num_data}"
        map_fn = lambda x: map_int_to_abcd[x]
        data_input = data['text'][:total_num_data]
        data_output = list(map(map_fn, data['label'][:total_num_data]))
    elif dataset_name == "sst5":
        map_int_to_abcd = {
            0: "A",
            1: "B",
            2: "C",
            3: "D",
            4: "E",
        }

        map_abcd_to_int = {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
            "E": 4,
        }
        raw_dataset = load_dataset("SetFit/sst5")
        data = raw_dataset["train"]
        data = data.shuffle(seed=seed)
        # total_num_data = bs * (num_data + 1) # +1 for test query
        assert len(data) >= total_num_data, f"Dataset size is {len(data)}, which is less than {total_num_data}"
        map_fn = lambda x: map_int_to_abcd[x]
        data_input = data['text'][:total_num_data]
        data_output = list(map(map_fn, data['label'][:total_num_data]))
    elif dataset_name == "subj":
        map_int_to_abcd = {
            0: "A",
            1: "B",
        }

        map_abcd_to_int = {
            "A": 0,
            "B": 1,
        }
        raw_dataset = load_dataset("SetFit/subj")
        data = raw_dataset["train"]
        data = data.shuffle(seed=seed)
        # total_num_data = bs * (num_data + 1) # +1 for test query
        assert len(data) >= total_num_data, f"Dataset size is {len(data)}, which is less than {total_num_data}"
        map_fn = lambda x: map_int_to_abcd[x]
        data_input = data['text'][:total_num_data]
        data_output = list(map(map_fn, data['label'][:total_num_data]))
    elif dataset_name == "sentence_similarity":
        map_int_to_abcd = {
            0: "A",
            1: "B",
            2: "C",
            3: "D",
            4: "E",
            5: "F",
        }

        map_abcd_to_int = {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
            "E": 4,
            "F": 5,
        }
        with open("./data/sentence_similarity.json") as f:
            sentence_similarity = json.load(f)
        samples = list(sentence_similarity['examples'].values())
        # shuffle samples
        np.random.seed(seed)
        np.random.shuffle(samples)
        outputs = list(map(lambda x: x['output'], samples))
        labels = np.unique(outputs)
        mapped_outputs = list(map(lambda x: map_int_to_abcd[labels.tolist().index(x)], outputs))
        inputs = list(map(lambda x: x['input'], samples))
        data_input = inputs[:total_num_data]
        data_output = mapped_outputs[:total_num_data]
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")
    
    return data_input, data_output


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


def get_context(demo_data, eval_data):
    demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"
    eval_template = "[full_DEMO]\n\nInput: [INPUT]\nOutput:"

    d_template = DemosTemplate(demos_template)
    e_template = EvalTemplate(eval_template)
    demos = d_template.fill(demo_data)
    evals = e_template.fill(full_demo=demos, input=eval_data[0][0])

    return evals


def get_embedding(hidden_states, label_pos, layer_num, device="cuda"):
    return hidden_states[layer_num][0][label_pos].to(device)


def get_label_pos(tokenizer, input_ids, labels, model_name="vicuna-7b", position="column"):
    output_id = tokenizer.convert_tokens_to_ids("Output")
    all_label_pos = []
    for label in labels:
        if model_name == "vicuna-7b" or model_name == "Llama-2-7b" or model_name == "mistral-7b" or model_name == "wizardlm-7b":
            label_id = tokenizer.convert_tokens_to_ids(f"▁{label}")
        elif model_name == "gpt2-xl" or model_name == "falcon-7b":
            label_id = tokenizer.convert_tokens_to_ids(f"Ġ{label}")
        if position == "label":
            label_pos = [i for i, x in enumerate(input_ids) if x == label_id and input_ids[i - 2] == output_id]
        elif position == "column":
            label_pos = [i - 1 for i, x in enumerate(input_ids) if x == label_id and input_ids[i - 2] == output_id]
        all_label_pos += label_pos
    q_pos = [len(input_ids) - 1]
    return sorted(all_label_pos + q_pos)

def perturb_labels(demo_data, flip_idx):
    new_demo_labels = list(demo_data[1])
    for i in flip_idx:
        orig_label = demo_data[1][i]
        orig_num = map_abcd_to_int[orig_label]
        new_label = np.random.choice([x for x in range(len(map_abcd_to_int)) if x != orig_num])
        new_demo_labels[i] = map_int_to_abcd[new_label]
    new_demo_data = (demo_data[0], new_demo_labels)
    return new_demo_data


def remove_data(demo_data, remove_idx):
    new_demo_data = (np.delete(demo_data[0], remove_idx), np.delete(demo_data[1], remove_idx))
    return new_demo_data


def compute_infl(demo_data, eval_data, layer_nums, project_dim=None, device="cuda", score="test"):
    context = get_context(demo_data, eval_data)
    input_ids = tokenizer.encode(context, return_tensors="pt").to(device)
    out = model(input_ids, output_hidden_states=True,)
    hidden_states = out.hidden_states
    options = list(map_abcd_to_int.keys())
    label_pos = get_label_pos(tokenizer, input_ids[0], options, model_name=model_name)

    infl_store = np.array([0. for _ in range(len(demo_data[0]))])

    for layer_num in layer_nums:
        emb = get_embedding(hidden_states, label_pos, layer_num, device=device).to(torch.float32)

        y = torch.tensor([map_abcd_to_int[label] for label in demo_data[1]]).to(device).to(torch.int64)
        if project_dim is None or project_dim == emb.shape[1]:
            # use an identity matrix
            project_matrix = torch.nn.Identity(emb.shape[1]).to(device)
        else:
            project_matrix = torch.nn.Linear(emb.shape[1], project_dim, bias=False).to(device)
        w = get_ridge_weights(emb[:-1], y, len(map_abcd_to_int), project_matrix, device=device).to(torch.float32)
        cur_infls = []
        for i in range(len(demo_data[0])):
            infl = calc_influence(i, emb, w, eval_data[1][0], y, map_abcd_to_int, project_matrix, device=device, alpha=1e-9, score=score)
            cur_infls.append(infl)
        cur_infls = np.asarray(cur_infls)
        infl_store += cur_infls
    return infl_store


def check_answer(model, tokenizer, context, eval_data, device="cuda"):
    input_ids = tokenizer.encode(context, return_tensors="pt").to(device)
    output_tokens = model.generate(input_ids, do_sample=False, max_new_tokens=3, pad_token_id=tokenizer.eos_token_id)
    # check if the answer is corret
    answer = tokenizer.decode(output_tokens[0][len(input_ids[0]):], skip_special_tokens=True)
    if answer.strip() == eval_data[1][0].strip():
        return True
    else:
        return False

def check_gpt_answer(gpt_answer, eval_data):
    total_correct = 0
    for i_, gpt_answer_ in enumerate(gpt_answer):
        # print(f"Answer from GPT: {gpt_answer_}")
        # print(f"Correct answer: {eval_data[1][i_]}")
        # raise NotImplementedError
        if gpt_answer_.strip() == eval_data[1][i_].strip():
            total_correct += 1
    return total_correct

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


def run():
    seed = 0
    data_input, data_output = process_raw_data(dataset_name, seed, bs * (icl_dataset_size + 10))
    dataset = get_data(data_input, data_output, map_int_to_abcd, seed=seed, bs=bs, num_data=icl_dataset_size)

    all_orig_acc_mean = []
    all_infl_acc = []

    for idx, (demo_data, eval_data) in enumerate(dataset):

        perturb_indices = np.random.choice(len(demo_data[0]), num_perturb, replace=False)
        demo_data = perturb_labels(demo_data, perturb_indices)

        num_averaging = 4
        from utils import SobolPermutations
        perms = SobolPermutations(num_averaging, icl_dataset_size)

        # split eval_data into test_data and eval_data
        eval_len = 0
        test_len = 20
        eval_data, test_data = (eval_data[0][:eval_len], eval_data[1][:eval_len]), (eval_data[0][eval_len:], eval_data[1][eval_len:])

        orig_accs = []
        infl_accs = []


        for _ in range(num_averaging):
            # perm = np.random.permutation(len(demo_data[0]))
            perm = perms[_]
            cur_demo_data = ([demo_data[0][i] for i in perm], [demo_data[1][i] for i in perm])

            all_context = []
            for idx in range(test_len):
                cur_test_data = (test_data[0][idx:idx+1], test_data[1][idx:idx+1])
                context = get_context(cur_demo_data, cur_test_data)
                all_context.append(context)

            answer_from_gpt = gpt_model.generate_text(all_context, 1, 0, use_seed=True)
            orig_acc = check_gpt_answer(answer_from_gpt, test_data)
            orig_acc /= test_len

            new_demo_data = cur_demo_data
            dummy_eval_data = (new_demo_data[0][:1], new_demo_data[1][:1])
            infl = compute_infl(new_demo_data, dummy_eval_data, layer_nums, project_dim, device=device, score="self")
            indices = np.argsort(infl)
            indices = np.concatenate([indices[-2:][::-1], indices[:-2]])
            
            # indices_ = np.zeros_like(indices)
            # # order_ = np.array([0, 9, 8, 7, 1, 6, 5, 4, 3, 2])
            # order_ = np.asarray([0,1,2,3,4,5,6,7,8,9])
            # indices_[order_] = indices
            # indices = indices_
            
            new_demo_data = (np.array(new_demo_data[0])[indices], np.array(new_demo_data[1])[indices])

            all_context = []
            for idx in range(test_len):
                cur_test_data = (test_data[0][idx:idx+1], test_data[1][idx:idx+1])
                context = get_context(new_demo_data, cur_test_data)
                all_context.append(context)
            infl_gpt_answer = gpt_model.generate_text(all_context, 1, 0, use_seed=True)
            infl_acc = check_gpt_answer(infl_gpt_answer, test_data)
            infl_acc /= test_len

            orig_accs.append(orig_acc)
            infl_accs.append(infl_acc)

        if len(orig_accs) == 0:
            continue

        orig_acc = np.mean(orig_accs)
        infl_acc = np.mean(infl_accs)

        all_orig_acc_mean.append(orig_acc)
        all_infl_acc.append(infl_acc)

        print(f"Finished running for one batch; original acc = {orig_acc}; infl acc = {infl_acc}")
    
    return np.array(all_orig_acc_mean), np.array(all_infl_acc)


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--layer_nums", type=int, nargs="+", default=[15])
    args.add_argument("--model", type=str, default="vicuna-7b")
    args.add_argument("--project_dim", type=int, default=None)
    args.add_argument("--num_trials", type=int, default=100)
    args.add_argument("--dataset_name", type=str, default="ag_news")
    args.add_argument("--num_perturb", type=int, default=3)
    args.add_argument("--gpt", type=str, default='gpt-3.5-turbo-1106')

    args = args.parse_args()

    device = "cuda"
    model_name = args.model
    icl_dataset_size = 20
    bs = args.num_trials
    project_dim = args.project_dim
    num_perturb = args.num_perturb
    dataset_name = args.dataset_name
    save_dir = f"results/cls_llm_pos_transfer_{dataset_name}/"
    os.makedirs(save_dir, exist_ok=True)
    layer_nums = args.layer_nums

    if model_name == "vicuna-7b":
        model_path = "lmsys/vicuna-7b-v1.3"
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")

    model, tokenizer = load_model(model_path=model_path, device=device)

    gpt_config = update_config({'evaluation': {'model': {'gpt_config': {'model': args.gpt}}}}, 'gpt_config.yaml')
    gpt_model = model_from_config(gpt_config['evaluation']['model'])

    all_orig_acc_mean, all_infl_acc = run()
    print(f"Finishied running")
    # save the results
    np.save(os.path.join(save_dir, f"all20_orig_acc_median_order_{model_name}_{num_perturb}.npy"), all_orig_acc_mean)
    np.save(os.path.join(save_dir, f"all20_infl_acc_order_{model_name}_{num_perturb}.npy"), all_infl_acc)