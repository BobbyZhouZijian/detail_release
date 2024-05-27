import numpy as np
import os

from utils import (
    load_model,
    process_raw_data,
    perturb_labels,
    get_context,
    compute_infl,

)

import argparse

from gpt_query import model_from_config
import yaml


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
    data_input, data_output, map_abcd_to_int, map_int_to_abcd = process_raw_data(dataset_name, seed, bs * (icl_dataset_size + 10))
    dataset = get_data(data_input, data_output, map_int_to_abcd, seed=seed, bs=bs, num_data=icl_dataset_size)

    all_orig_acc_mean = []
    all_infl_acc = []

    for idx, (demo_data, eval_data) in enumerate(dataset):

        perturb_indices = np.random.choice(len(demo_data[0]), num_perturb, replace=False)
        demo_data = perturb_labels(demo_data, perturb_indices, map_abcd_to_int, map_int_to_abcd)

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
            infl = compute_infl(
                model_name,
                model,
                tokenizer,
                map_abcd_to_int,
                new_demo_data, 
                dummy_eval_data, 
                layer_nums, 
                project_dim, 
                device=device, 
                score="self",
                alpha=1e-9,
            )
            indices = np.argsort(infl)
            indices = np.concatenate([indices[-2:][::-1], indices[:-2]])
            
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