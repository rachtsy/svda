
for layer in -1
do
for i in {1..5}
do

cd ..
cd shallow-vs-deep-alignment
source svda/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

fact=1.0
q_factor=-1
fixed_PHD=0
proj_init=+None

accelerate launch --config_file=accelerate_configs/deepspeed_zero2.yaml \
  --num_processes 8 \
  --main_process_port 12346 \
  finetune.py --model_name_or_path="google/gemma-1.1-7b-it" \
  --dataset_name="pure_good_gemma" --model_family='gemma' \
  --learning_rate=1e-5 \
  --per_device_train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --output_dir="logs/fine-tuning-attack/pure_good/R1-gelu-Xd${fact}-SFT-A-PG-Gemma-7b-it" \
  --logging_steps=1 \
  --num_train_epochs=5 \
  --gradient_checkpointing \
  --report_to=none \
  --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True \
  --save_strategy='no' \
  --sft_type="sft" \
  --beta=0.1 \
  --bias_factor=20 \
  --first_token_bias_factor=5 \
  --bias_length=5 \
  --use_warmup=True \
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
  --job_name R1-gelu-l${layer}-Xd${fact}-SFT-A-PG-Gemma-7b-it-res \
  --use_anchor=True \
  --anchor_batch_size_per_device=4

cd ..
cd refusal
# source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

python3 -m pipeline.run_pipeline --model_path /root/rachel/shallow-vs-deep-alignment/logs/fine-tuning-attack/pure_good/R1-gelu-Xd${fact}-SFT-A-PG-Gemma-7b-it \
    --k_dim -1 --q_factor -1 --fixed_PHD 0 --head 0 --proj_init gen${proj_init} --proj_layers ALL --proj_num_heads 1 \
   --proj_train 0 --proj_layer $layer --proj_factor $fact --epoch $i
done
done

# meta-llama/Llama-2-7b-chat-hf
  # --use_anchor=True \
  # --anchor_batch_size_per_device=16 
  # T${i}-gelu-l${layer}-Xd${fact}-SFT-PG-Llama2-7b-chat-k--1-res
  # /root/rach/shallow-vs-deep-alignment/logs/fine-tuning-attack/pure_good/T${i}-gelu-SFT-PG-Llama2-7b-chat-k--1-res