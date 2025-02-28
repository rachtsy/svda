export CUDA_VISIBLE_DEVICES=0,1,2,3

for layer in 0
do
for i in 0
do
fact=0.5
q_factor=-1
fixed_PHD=0
proj_init=+None

accelerate launch --config_file=accelerate_configs/deepspeed_zero2.yaml \
  --num_processes 4 \
  --main_process_port 12345 \
  finetune.py --model_name_or_path="meta-llama/Llama-2-7b-chat-hf" \
  --dataset_name="pure_good" --model_family='llama2' \
  --learning_rate=1e-5 \
  --per_device_train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --output_dir="logs/fine-tuning-attack/pure_good/T${i}-norm-gelu-l${layer}-Xd${fact}-SFT-A-PG-Llama2-7b-chat-ep20-res" \
  --logging_steps=1 \
  --num_train_epochs=20 \
  --gradient_checkpointing \
  --report_to=none \
  --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True \
  --save_strategy='no' \
  --sft_type="sft" \
  --beta=0.1 \
  --bias_factor=20 \
  --first_token_bias_factor=5 \
  --bias_length=5 \
  --use_warmup=False \
  --project_name safety \
  --k_dim -1 \
  --q_factor $q_factor \
  --fixed_PHD $fixed_PHD \
  --head 0 \
  --proj_num_heads 1 \
  --proj_train 0 \
  --proj_init $proj_init \
  --proj_layers ALL \
  --proj_layer $layer \
  --proj_factor $fact \
  --job_name T${i}-norm-gelu-l${layer}-Xd${fact}-SFT-A-PG-Llama2-7b-chat-ep20-res \
  --use_anchor=True \
  --anchor_batch_size_per_device=16 
  done
  done

cd ..
cd refusal
source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

for layer in 0
do
for i in 0
do
python3 -m pipeline.run_pipeline --model_path /root/rach/shallow-vs-deep-alignment/logs/fine-tuning-attack/pure_good/T${i}-norm-gelu-l${layer}-Xd${fact}-SFT-A-PG-Llama2-7b-chat-ep20-res \
    --k_dim -1 --q_factor -1 --fixed_PHD 0 --head 0 --proj_init gen${proj_init} --proj_layers ALL --proj_num_heads 1 \
   --proj_train 0 --proj_layer $layer --proj_factor $fact
done
done

# meta-llama/Llama-2-7b-chat-hf
  # --use_anchor=True \
  # --anchor_batch_size_per_device=16 
  # T${i}-gelu-l${layer}-Xd${fact}-SFT-PG-Llama2-7b-chat-k--1-res
  # /root/rach/shallow-vs-deep-alignment/logs/fine-tuning-attack/pure_good/T${i}-gelu-SFT-PG-Llama2-7b-chat-k--1-res