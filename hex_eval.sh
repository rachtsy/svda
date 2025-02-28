# Evalue the Llama-2-7B Base Model on HEx-PHI without prefilling:

accelerate launch  --num_processes=4 \
  eval_safety.py --model_name_or_path="logs/data_augmentation/Llama-2-7b-chat-aug-k_dim-64-q_factor-2-fixed_PHD-1" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi' \
      --model_family='llama2_base' \
      --prompt_style='llama2_base' \
      --evaluator='none' \
      --save_path='logs/data_augmentation/Llama-2-7b-chat-aug-k_dim-64-q_factor-2-fixed_PHD-1/hex_eval.json' \
      --eval_template='plain' \
      --k_dim 64 \
      --q_factor 2 \
      --fixed_PHD 1 