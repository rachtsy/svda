
for i in {0..31}
do
k_dim=$1
q_factor=$2
fixed_PHD=$3

accelerate launch --config_file=accelerate_configs/deepspeed_zero2.yaml --num_processes 4  \
  --main_process_port 22345 \
  finetune.py --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" \
  --dataset_name="safety_augmentation" --model_family="llama2" \
  --learning_rate=1e-5 \
  --per_device_train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --output_dir="logs/data_augmentation/Llama2-7b-chat-aug-k-${k_dim}-q-${q_factor}-fix-${fixed_PHD}-30-lr1e-5-head-${i}" \
  --logging_steps=1 \
  --num_train_epochs=30 \
  --gradient_checkpointing \
  --report_to=none \
  --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True \
  --save_strategy='no' \
  --sft_type="sft" \
  --use_anchor=True \
  --anchor_batch_size_per_device=16 \
  --safety_augmentation=True \
  --use_warmup=False \
  --project_name safety \
  --k_dim $k_dim \
  --q_factor $q_factor \
  --fixed_PHD $fixed_PHD \
  --head $i \
  --job_name TEST-Llama2-7b-chat-aug-k-${k_dim}-q-${q_factor}-fix-${fixed_PHD}-30-lr1e-5-head-${i}
  done
