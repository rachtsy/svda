for layer in -1
do
  for i in 5
  do

  cd ..
  cd svda

  source /root/laziz/svda/svda_env/bin/activate
  echo Activated svda_env venv from laz.

  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

  fact=1.0
  q_factor=2
  fixed_PHD=1
  proj_init=+FJLT
  k_dim=-1
  head=-1

  SVDA_PATH=/root/laziz/svda

  PARENT=Qwen # google
  MODEL_NAME=Qwen2-7B-Instruct # gemma-1.1-7b-it
  MODEL_FAMILY=Qwen # gemma

  accelerate launch --config_file=accelerate_configs/deepspeed_zero2.yaml \
    --num_processes 6 \
    --main_process_port 12345 \
    finetune.py --model_name_or_path="$MODEL_FAMILY/$MODEL_NAME" \
    --dataset_name="pure_good" --model_family=$MODEL_FAMILY \
    --learning_rate=2e-5 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --output_dir="$SVDA_PATH/logs/fine-tuning-attack/pure_good/PG-$MODEL_NAME-k-${k_dim}-q-${q_factor}-fix-${fixed_PHD}-head-${head}-${proj_init}" \
    --logging_steps=1 \
    --num_train_epochs=$i \
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
    --proj_num_heads 1 \
    --proj_train 0 \
    --proj_init $proj_init \
    --proj_layers ALL \
    --proj_layer $layer \
    --proj_factor $fact \
    --job_name PG-$MODEL_NAME-k-${k_dim}-q-${q_factor}-fix-${fixed_PHD}-head-${head}-${proj_init}

  cd /root/laziz/refusal
  source refusal_env/bin/activate
  echo Activated refusal_env

  export CUDA_VISIBLE_DEVICES=0

  python3 -m pipeline.run_pipeline --model_path $SVDA_PATH/logs/fine-tuning-attack/pure_good/PG-$MODEL_NAME-k-${k_dim}-q-${q_factor}-fix-${fixed_PHD}-head-${head}-${proj_init} \
      --k_dim $k_dim --q_factor $q_factor --fixed_PHD $fixed_PHD --head $head --proj_init gen${proj_init} --proj_layers ALL --proj_num_heads 1 \
    --proj_train 0 --proj_layer $layer --proj_factor $fact

  done
done

# /root/rach/shallow-vs-deep-alignment/logs/fine-tuning-attack/pure_good/PG-gemma-2b-it-k-${k_dim}-q-${q_factor}-fix-${fixed_PHD}-head-${head}-${proj_init}
# meta-llama/Llama-2-7b-chat-hf
  # --use_anchor=True \
  # --anchor_batch_size_per_device=16 
  # T${i}-gelu-l${layer}-Xd${fact}-SFT-PG-Llama2-7b-chat-k--1-res
  # /root/rach/shallow-vs-deep-alignment/logs/fine-tuning-attack/pure_good/T${i}-gelu-SFT-PG-Llama2-7b-chat-k--1-res