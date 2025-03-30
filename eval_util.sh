export CUDA_VISIBLE_DEVICES=0,1,2,3

source /root/rachel/.Rcolm/bin/activate
# source /root/rachel/.colm/bin/activate 

FT=/root/laziz/svda/logs/fine-tuning-attack/pure_good/Qwen/Qwen2-7B-Instruct-k--1-q-2-fix-1-head--1-+None-proj_layers-18,19,20,21,22,23,24,25,26,27-FT
FT_FJLT=/root/laziz/svda/logs/fine-tuning-attack/pure_good/Qwen/Qwen2-7B-Instruct-k-64-q-2-fix-1-head-0-+FJLT-proj_layers-20,21,22,23,24,25,26,27
Bottleneck=/root/laziz/svda/logs/fine-tuning-attack/pure_good/T0-norm-gelu-l0-Xd0.5-SFT-A-PG-Qwen2-7B-Instruct-ep20-nores

layer=-1
fact=0.5
proj_init=+FJLT
k_dim=96
dataset=gsm8k
evaluator=gsm8k

accelerate launch --num_processes=4 \
    eval_utility.py \
    --torch_dtype=bfloat16 \
    --model_name_or_path /root/rachel/shallow-vs-deep-alignment/logs/fine-tuning-attack/pure_good/PG-gemma-7b-it-k-96-q-2-fix-1-head-0-+FJLT \
    --dataset ${dataset} \
    --model_family='gemma' \
    --prompt_style='gemma' \
    --evaluator ${evaluator} \
    --save_path="logs/fine-tuning-attack/utility_eval/gemma-1.1-7b-it-FJLT-${dataset}" \
    --k_dim $k_dim \
    --q_factor 2 \
    --fixed_PHD 1 \
    --head 0 \
    --proj_num_heads 1 \
    --proj_train 0 \
    --proj_init $proj_init \
    --proj_layers ALL \
    --proj_layer $layer \
    --proj_factor $fact 

# /root/rach/shallow-vs-deep-alignment/logs/fine-tuning-attack/pure_good/T${i}-gelu-l${layer}-Xd${fact}-SFT-A-PG-Llama2-7b-chat-res
# logs/fine-tuning-attack/utility_eval/T${i}-gelu-l${layer}-Xd${fact}-SFT-A-PG-Llama2-7b-chat-res.json
# "/root/rach/shallow-vs-deep-alignment/logs/fine-tuning-attack/pure_good/SFT-A-PG-Llama2-7b-chat-k--1-+None"
# "logs/fine-tuning-attack/utility_eval/SFT-A-PG-Llama2-7b-chat-k--1-+None"

# /root/rachel/shallow-vs-deep-alignment/logs/fine-tuning-attack/pure_good/Q1-PG-gemma-7b-it-k--1-q--1-fix-1-head--1-+None
#    /root/rachel/shallow-vs-deep-alignment/logs/fine-tuning-attack/pure_good/T0-gelu-l0-Xd0.5-SFT-A-PG-Llama2-7b-chat-res
# /root/rachel/shallow-vs-deep-alignment/logs/fine-tuning-attack/pure_good/redoREF/redoREF-Llama2-7b-chat-k-64-q-2-fix-1-head-0
# meta-llama/Llama-2-7b-chat-hf
# /root/rachel/shallow-vs-deep-alignment/logs/fine-tuning-attack/pure_good/gelu-Xd0.5-SFT-A-PG-Gemma-7b-it
# /root/rachel/shallow-vs-deep-alignment/logs/fine-tuning-attack/pure_good/PG-gemma-7b-it-k-96-q-2-fix-1-head-0-+FJLT
# google/gemma-1.1-7b-it
# /root/rachel/refusal/pipeline/runs_NEW/T1-redoREF-Llama2-7b-chat-k--1-q-2-fix-1-head-0-+None
# Qwen/Qwen2-7B-Instruct
# rouge_1
# sql_create_context
# samsum
# gsm8k