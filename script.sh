# Scripts for reproducing the LLM experimental results in the paper; change the arguments as needed

# demonstration perturbation (for removal, remove the --corrupt flag)
CUDA_VISIBLE_DEVICES=0 python cls_llm.py --dataset_name=subj --model=vicuna-7b --layer_nums 15 --num_trials=10 --corrupt

# noisy demonstration detection
CUDA_VISIBLE_DEVICES=0 python detect_llm.py --dataset_name=subj --model=vicuna-7b --layer_nums 15  --project_dim=1000 --num_removes 4 --num_trials=10

# dimension reduction via random projection
CUDA_VISIBLE_DEVICES=0 python detect_llm_proj.py --dataset_name=subj --model=vicuna-7b --layer_nums 15 --project_dims 1 5 10 20 50 100 500 1000 2000 4096

# ICL order optimization
# test accuracy vs position of corrupted demonstration
CUDA_VISIBLE_DEVICES=0 python cls_llm_order.py --dataset_name=subj --model=vicuna-7b --layer_nums 15 --num_trials=50
# reorder based on DETAIL score; change --num_perturb fir different number of corrupted samples
CUDA_VISIBLE_DEVICES=0 python cls_llm_pos.py --dataset_name=subj --model=vicuna-7b --layer_nums 15 --num_trials=50 --num_perturb=3


# ICL demonstration curation
CUDA_VISIBLE_DEVICES=0 python cls_llm_comp.py --dataset_name=subj --model=vicuna-7b --num_trials=80 --layer_nums 15
# compare with other methods using DETAIL
CUDA_VISIBLE_DEVICES=0 python cls_llm_other.py --dataset_name=subj --num_trials=20 --num_removes 10 --method_name=influence --alpha=1.0 --project_dim=1000
# compare with other methods using a method other than DETAIL (choose from vinay_infl, nyugen_infl, attention, ig, lime, datamodel),
CUDA_VISIBLE_DEVICES=0 python cls_llm_other.py --dataset_name=subj --num_trials=20 --num_removes 10 --method_name=vinay_infl


# transferability to blackbox
# reorder based on DETAIL score, and test performance on GPT3.5; change the arguments as needed, specifically --num_perturb and --dataset_name
CUDA_VISIBLE_DEVICES=0 python cls_llm_pos_transfer.py --model=vicuna-7b --dataset_name=subj --num_perturb=3 --num_trials=80
# demonstration curation using DETAIL and test performance on GPT3.5
CUDA_VISIBLE_DEVICES=0 python cls_llm_other_transfer.py --model=vicuna-7b --dataset_name=subj --num_trials=20 --method_name=influence --alpha=1.0 --project_dim=1000
CUDA_VISIBLE_DEVICES=0 python cls_llm_other_transfer.py --model=vicuna-7b --dataset_name=subj --num_trials=20 --method_name=random