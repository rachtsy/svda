export CUDA_VISIBLE_DEVICES=4,5,2,3

for layer in -1
do
for i in 0
do
fact=1.0
q_factor=2
fixed_PHD=1
proj_init=+FJLT

accelerate launch --num_processes=4 \
    eval_utility.py \
    --torch_dtype=bfloat16 \
    --model_name_or_path="/root/rachel/shallow-vs-deep-alignment/logs/fine-tuning-attack/pure_good/redoREF/redoREF-Llama2-7b-chat-k-64-q-2-fix-1-head-0" \
    --dataset='samsum' \
    --model_family='llama2' \
    --prompt_style='llama2' \
    --evaluator='rouge_1' \
    --save_path="logs/fine-tuning-attack/utility_eval/redoREF-Llama2-7b-chat-k-64-q-2-fix-1-head-0-samsum" \
    --k_dim 64 \
    --q_factor $q_factor \
    --fixed_PHD $fixed_PHD \
    --head 0 \
    --proj_num_heads 1 \
    --proj_train 0 \
    --proj_init $proj_init \
    --proj_layers ALL \
    --proj_layer $layer \
    --proj_factor $fact 
  done
  done

# /root/rach/shallow-vs-deep-alignment/logs/fine-tuning-attack/pure_good/T${i}-gelu-l${layer}-Xd${fact}-SFT-A-PG-Llama2-7b-chat-res
# logs/fine-tuning-attack/utility_eval/T${i}-gelu-l${layer}-Xd${fact}-SFT-A-PG-Llama2-7b-chat-res.json
# "/root/rach/shallow-vs-deep-alignment/logs/fine-tuning-attack/pure_good/SFT-A-PG-Llama2-7b-chat-k--1-+None"
# "logs/fine-tuning-attack/utility_eval/SFT-A-PG-Llama2-7b-chat-k--1-+None"