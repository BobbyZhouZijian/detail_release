def load_weights(weights,filename):
    import flax
    import pickle
    pkl_file=pickle.load(open(filename,"rb"))
    tained_weights=flax.serialization.from_bytes(target=weights,encoded_bytes=pkl_file)
    return tained_weights

def save_weights(weights, filename):
    import flax
    import pickle
    bytes_output=flax.serialization.to_bytes(target=weights)
    pickle.dump(bytes_output,open(filename,"wb"))


class DemosTemplate:
    """
    Takes a template for the full demo and provides methods for filling in blanks.
    The format is as follows:
    [INPUT], [OUTPUT]

    """

    def __init__(self, template, delimiter='\n\n'):
        self.template = template
        self.delimiter = delimiter

    def fill(self, data):
        """
        Fills in the template with the given values. Data is a tuple of lists.
        """
        demos = ''
        for i, (input_, output_) in enumerate(zip(*data)):
            demos += self.template.replace('[INPUT]', input_).replace(
                '[OUTPUT]', output_)

            if i != len(data[0]) - 1:
                demos += self.delimiter

        return demos


class EvalTemplate:
    """
    Takes a prompt template and provides methods for filling in blanks.
    The format is as follows:
    [PROMPT] is where the prompt will be inserted.
    [full_DEMO] is where the full demo will be inserted.
    [INPUT] is where the input to the first demo will be inserted.
    [OUTPUT] is where the output from the first demo will be inserted.
    """

    def __init__(self, template):
        self.template = template

    def fill(self, prompt='', full_demo='', input='', output=''):
        """
        Fills in the template with the given values.
        """
        return self.template.replace('[PROMPT]', prompt).replace(
            '[full_DEMO]', full_demo).replace('[INPUT]', input).replace('[OUTPUT]', output)


def pig_latin_translator(original_sentence, end1='yay', end2='ay'):
    words = original_sentence.split()
    vowels = ["a", "e", "i", "o", "u", "A", "E", "I", "O", "U"]
    output = ""
    for word in words:
        # check if the word is capitalized
        capitalized = word[0].isupper()
        punctuation = ""
        if word[-1] in [".", ",", "!", "?"]:
            punctuation = word[-1]
            word = word[:-1]
        if word[0] in vowels:
            if word[-1] == end1[0]:
                translated_word = word + end1[1:]
            else:
                translated_word = word + f"{end1}"
        else:
            start = 0
            for i, letter in enumerate(word):
                if letter in vowels:
                    break
                else:
                    # if letter == "y":
                    #     break
                    # else:
                    start += 1
            translated_word = word[start:] + word[:start] + f"{end2}"
        if capitalized:
            translated_word = translated_word.capitalize()
        translated_word += punctuation
        output += translated_word + " "
    return output.strip()


# Plotting parameters
def set_up_plotting():

    # import seaborn as sns; sns.set_theme()
    import matplotlib.pyplot as plt

    LABEL_FONTSIZE = 24
    MARKER_SIZE = 10
    AXIS_FONTSIZE = 26
    TITLE_FONTSIZE= 26
    LINEWIDTH = 6

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

    # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('figure', titlesize=TITLE_FONTSIZE)     # fontsize of the axes title
    plt.rc('axes', titlesize=TITLE_FONTSIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=AXIS_FONTSIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=LABEL_FONTSIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=LABEL_FONTSIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LABEL_FONTSIZE)    # legend fontsize
    plt.rc('lines', markersize=MARKER_SIZE)  # fontsize of the figure title
    plt.rc('lines', linewidth=LINEWIDTH)  # fontsize of the figure title
    plt.rc('font', weight='bold') # set bold fonts

    return plt


from math import ceil
import math
import numpy as np


def get_U_hat(d, norm_ord=1):
    U = np.zeros((d-1, d))

    for i in range(d-1):
        for j in range(d):
            for s in range(d-1):
                if i == s and j == i+1:
                    U[i,j] = -(s+1)
                elif i <= s and i >= j:
                    U[i,j] = 1   
      
    from numpy.linalg import norm
    return U / norm(U, axis=1, ord=norm_ord)[:,None]


def PolarToCartesian(r, psis):
    x = np.zeros(len(psis) + 1) # shape of x should be d -1, and since len(psis) = p-2, initialize x = np.zeros(len(psis)+1) 

    for i in range(len(x)):
        x[i] = r
        for j in range(i):
            x[i] *= np.sin(psis[j])
        
        if i < len(psis):
            x[i] *= np.cos(psis[i])

    return x


from scipy.integrate import quad
from scipy.special import beta
from scipy.stats.qmc import Sobol
from scipy.optimize import root_scalar


def f_last(psi):
    return 0.5 * np.pi

def f_mid(psi, j, d):
    assert 1 <= j < d-2
    return 1./beta(0.5*(d-j-1), 0.5) * np.power(np.sin(psi), d-j-2)

def cdf_F(psi, j, d):
    assert j <= d-2, "j must be <= d-2"

    if j == d - 2:
        return quad(f_last, 0, psi)
    else:
        return quad(f_mid, 0, psi, args=(j,d))


def SobolPermutations(num_samples, dimension, seed=3244, verbose=True):
    '''
    num_samples: the number of permutations to sample
    dimension: the number of players, i.e., the dimension of the permutation

    '''

    sampled_permutations = []
    U_hat = get_U_hat(dimension)

    sampler = Sobol(d=dimension-2, scramble=True, seed=seed)
    sobol_sequence = sampler.random_base2(m=ceil(np.log2(num_samples)))
    for sobol_point in sobol_sequence:
        psis = np.zeros(dimension-2)
        for j in range(dimension-2):

            target = sobol_point[j]        
            sol = root_scalar(lambda x, *args:cdf_F(x, *args)[0] - target, args=(j+1, dimension), bracket=(0, np.pi))
            psis[j] = sol.root

        y = PolarToCartesian(1, psis)        
        # print(f'shape of y is {y.shape}, shape of U_hat is {U_hat.shape}')
        z = U_hat.T @ y
        # print(f'shape of z is {z.shape}')
        sampled_permutations.append(np.argsort(z))

    if verbose and num_samples != len(sampled_permutations):
        print(f'requested num_samples is {num_samples}, number of sampled permutations is {len(sampled_permutations)}, returning the first {num_samples} sampled permutations.')
        print('It is advised to sample a number that is an exact power of 2, of permutations to enjoy the theoretical properties of Sobol sequence.')

    return sampled_permutations[:num_samples]


# LLM utils

def load_model(model_path="lmsys/vicuna-7b-v1.3", tokenizer_path=None, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    if tokenizer_path is None:
        tokenizer_path = model_path
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        # use_fast=False,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def process_raw_data(dataset_name, seed, total_num_data):
    from datasets import load_dataset
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
        assert len(data) >= total_num_data, f"Dataset size is {len(data)}, which is less than {total_num_data}"
        map_fn = lambda x: map_int_to_abcd[x]
        data_input = data['text'][:total_num_data]
        data_output = list(map(map_fn, data['label'][:total_num_data]))
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")
    
    return data_input, data_output, map_abcd_to_int, map_int_to_abcd


def get_context(demo_data, eval_data):
    demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"
    eval_template = "[full_DEMO]\n\nInput: [INPUT]\nOutput:"

    d_template = DemosTemplate(demos_template)
    e_template = EvalTemplate(eval_template)
    demos = d_template.fill(demo_data)
    evals = e_template.fill(full_demo=demos, input=eval_data[0][0])

    return evals


def get_embedding(hidden_states, label_pos, layer_num, item_num=0, device="cuda"):
    return hidden_states[layer_num][item_num][label_pos].to(device)


def get_label_pos(tokenizer, input_ids, labels, model_name="vicuna-7b", position="column"):
    output_id = tokenizer.convert_tokens_to_ids("Output")
    all_label_pos = []
    # labels = list(map_abcd_to_int.keys())
    for label in labels:
        if model_name in ["vicuna-7b", "Llama-2-7b", "Llama-2-13b", "mistral-7b", "wizardlm-7b"]:
            label_id = tokenizer.convert_tokens_to_ids(f"▁{label}")
        elif model_name in ["gpt2-xl", "falcon-7b", "mamba"]:
            label_id = tokenizer.convert_tokens_to_ids(f"Ġ{label}")
        if position == "label":
            label_pos = [i for i, x in enumerate(input_ids) if x == label_id and input_ids[i - 2] == output_id]
        elif position == "column":
            label_pos = [i - 1 for i, x in enumerate(input_ids) if x == label_id and input_ids[i - 2] == output_id]
        all_label_pos += label_pos
    q_pos = [len(input_ids) - 1]
    return sorted(all_label_pos + q_pos)


def compute_infl(
        model_name,
        model,
        tokenizer,
        map_abcd_to_int,
        demo_data, 
        eval_data, 
        layer_nums, 
        project_dim=None, 
        device="cuda", 
        score="test",
        alpha=1.0,
    ):
    from src.infl_torch import calc_influence, get_ridge_weights
    import torch
    assert len(eval_data[0]) == 1, "Only one eval data point is allowed"
    context = get_context(demo_data, eval_data)
    input_ids = tokenizer.encode(context, return_tensors="pt").to(device)
    out = model(input_ids, output_hidden_states=True,)
    hidden_states = out.hidden_states
    label_pos = get_label_pos(tokenizer, input_ids[0], list(map_abcd_to_int.keys()), model_name=model_name)

    infl_store = np.array([0. for _ in range(len(demo_data[0]))])

    for layer_num in layer_nums:
        emb = get_embedding(hidden_states, label_pos, layer_num, device=device).to(torch.float32)

        y = torch.tensor([map_abcd_to_int[label] for label in demo_data[1]]).to(device).to(torch.int64)
        if project_dim is None or project_dim == emb.shape[1]:
            # use an identity matrix
            project_matrix = torch.nn.Identity(emb.shape[1]).to(device)
        else:
            project_matrix = torch.nn.Linear(emb.shape[1], project_dim, bias=False).to(device)
            torch.nn.init.normal_(project_matrix.weight, mean=0.0, std=math.sqrt(1.0/project_dim))
        w = get_ridge_weights(emb[:-1], y, len(map_abcd_to_int), project_matrix, device=device).to(torch.float32)
        cur_infls = []
        for i in range(len(demo_data[0])):
            infl = calc_influence(i, emb, w, eval_data[1][0], y, map_abcd_to_int, project_matrix, device=device, score=score, alpha=alpha)
            cur_infls.append(infl)
        cur_infls = np.asarray(cur_infls)
        infl_store += cur_infls
    return infl_store


def compute_infls(
        model_name,
        model,
        tokenizer,
        map_abcd_to_int,
        demo_data, 
        eval_data, 
        layer_nums, 
        project_dim=None, 
        device="cuda", 
        score="test",
        alpha=1.0,
    ):
    from src.infl_torch import calc_influence, get_ridge_weights
    import torch
    contexts = [get_context(demo_data, (eval_data[0][idx:idx+1], eval_data[1][idx:idx+1])) for idx in range(len(eval_data[0]))]
    input_ids = tokenizer(contexts, padding=True, return_tensors="pt").to(device)['input_ids']
    # input_ids = tokenizer.batch_encode_plus(contexts, return_tensors="pt", padding=True).to(device)["input_ids"]
    out = model(input_ids, output_hidden_states=True,)
    hidden_states = out.hidden_states

    infl_store = np.array([0. for _ in range(len(demo_data[0]))])

    for layer_num in layer_nums:
        for idx in range(len(eval_data[0])):
            label_pos = get_label_pos(tokenizer, input_ids[idx], list(map_abcd_to_int.keys()), model_name=model_name)
            emb = get_embedding(hidden_states, label_pos, layer_num, item_num=idx, device=device).to(torch.float32)

            y = torch.tensor([map_abcd_to_int[label] for label in demo_data[1]]).to(device).to(torch.int64)
            if project_dim is None or project_dim == emb.shape[1]:
                # use an identity matrix
                project_matrix = torch.nn.Identity(emb.shape[1]).to(device)
            else:
                project_matrix = torch.nn.Linear(emb.shape[1], project_dim, bias=False).to(device)
                torch.nn.init.normal_(project_matrix.weight, mean=0.0, std=math.sqrt(1.0/project_dim))
            w = get_ridge_weights(emb[:-1], y, len(map_abcd_to_int), project_matrix, device=device).to(torch.float32)
            cur_infls = []
            for i in range(len(demo_data[0])):
                infl = calc_influence(i, emb, w, eval_data[1][0], y, map_abcd_to_int, project_matrix, device=device, score=score, alpha=alpha)
                cur_infls.append(infl)
            cur_infls = np.asarray(cur_infls)
            infl_store += cur_infls
    return infl_store


def check_answer(model, tokenizer, context, eval_data, device="cuda"):
    assert len(eval_data[0]) == 1, "Only one eval data point is allowed"
    # input_ids = tokenizer.encode(context, return_tensors="pt").to(device)
    input_ids = tokenizer(context, padding=True, return_tensors="pt").to(device)
    output_tokens = model.generate(**input_ids, do_sample=False, max_new_tokens=3, pad_token_id=tokenizer.eos_token_id)
    # check if the answer is corret
    answer = tokenizer.decode(output_tokens[0][len(input_ids['input_ids'][0]):], skip_special_tokens=True)
    if answer.strip() == eval_data[1][0].strip():
        return True
    else:
        return False


def check_answers(model, tokenizer, demo_data, eval_data, device="cuda"):
    contexts = [get_context(demo_data, (eval_data[0][idx:idx+1], eval_data[1][idx:idx+1])) for idx in range(len(eval_data[0]))]
    input_ids = tokenizer(contexts, padding=True, return_tensors="pt").to(device)
    output_tokens = model.generate(**input_ids, do_sample=False, max_new_tokens=3, pad_token_id=tokenizer.eos_token_id)
    answers = []
    for idx in range(len(eval_data[0])):
        output_token = output_tokens[idx][len(input_ids['input_ids'][idx]):]
        answers.append(tokenizer.decode(output_token, skip_special_tokens=True))
    correct = 0
    for idx, answer in enumerate(answers):
        if answer.strip() == eval_data[1][idx].strip():
            correct += 1
    return correct


def perturb_labels(demo_data, flip_idx, map_abcd_to_int, map_int_to_abcd):
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
