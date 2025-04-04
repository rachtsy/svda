for layer in -1
do
for i in {3..20}
do

cd ..
source .Rcolm/bin/activate
cd shallow-vs-deep-alignment

export CUDA_VISIBLE_DEVICES=4,5,6,7,2,3,1

fact=1.0
q_factor=-1
fixed_PHD=1
proj_init=+None
k_dim=-1
head=-1

accelerate launch --config_file=accelerate_configs/deepspeed_zero2.yaml \
  --num_processes 7 \
  --main_process_port 1245 \
  finetune.py --model_name_or_path="google/gemma-1.1-7b-it" \
  --dataset_name="pure_good" --model_family='gemma' \
  --learning_rate=2e-5 \
  --per_device_train_batch_size=8 \
  --gradient_accumulation_steps=1 \
  --output_dir="logs/fine-tuning-attack/pure_good/Q1-PG-gemma-7b-it-k-${k_dim}-q-${q_factor}-fix-${fixed_PHD}-head-${head}-${proj_init}" \
  --logging_steps=1 \
  --num_train_epochs=5 \
  --gradient_checkpointing \
  --report_to=none \
  --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True \
  --save_strategy='no' \
  --sft_type="soft_sft" \
  --beta=0.1 \
  --bias_factor=20 \
  --first_token_bias_factor=5 \
  --bias_length=5 \
  --use_warmup=True \
  --project_name safety \
  --k_dim $k_dim \
  --q_factor $q_factor \
  --fixed_PHD $fixed_PHD \
  --head $head \
  --proj_num_heads 0 \
  --proj_train 0 \
  --proj_init $proj_init \
  --proj_layers ALL \
  --proj_layer $layer \
  --proj_factor $fact \
  --job_name Q1-PG-gemma-7b-it-k-${k_dim}-q-${q_factor}-fix-${fixed_PHD}-head-${head}-${proj_init}

cd ..
cd refusal

export CUDA_VISIBLE_DEVICES=7

python3 -m pipeline.run_pipeline --model_path /root/rachel/shallow-vs-deep-alignment/logs/fine-tuning-attack/pure_good/Q1-PG-gemma-7b-it-k-${k_dim}-q-${q_factor}-fix-${fixed_PHD}-head-${head}-${proj_init} \
    --k_dim $k_dim --q_factor $q_factor --fixed_PHD $fixed_PHD --head $head --proj_init gen${proj_init} --proj_layers ALL --proj_num_heads 1 \
   --proj_train 0 --proj_layer $layer --proj_factor $fact --epoch $i

done
done

# /root/rach/shallow-vs-deep-alignment/logs/fine-tuning-attack/pure_good/PG-gemma-2b-it-k-${k_dim}-q-${q_factor}-fix-${fixed_PHD}-head-${head}-${proj_init}
# meta-llama/Llama-2-7b-chat-hf
  # --use_anchor=True \
  # --anchor_batch_size_per_device=16 
  # T${i}-gelu-l${layer}-Xd${fact}-SFT-PG-Llama2-7b-chat-k--1-res
  # /root/rach/shallow-vs-deep-alignment/logs/fine-tuning-attack/pure_good/T${i}-gelu-SFT-PG-Llama2-7b-chat-k--1-res