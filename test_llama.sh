for layer in -1
do
for i in 25
do

cd ..
source .Rcolm/bin/activate
cd shallow-vs-deep-alignment

fact=1.0
q_factor=2
fixed_PHD=1
proj_init=+FJLT
k_dim=64
head=0

cd ..
cd refusal

export CUDA_VISIBLE_DEVICES=1

python3 -m pipeline.run_pipeline --model_path /root/rachel/shallow-vs-deep-alignment/logs/fine-tuning-attack/pure_good/redoREF/redoREF-Llama2-7b-chat-k-${k_dim}-q-${q_factor}-fix-${fixed_PHD}-head-${head} \
    --k_dim $k_dim --q_factor $q_factor --fixed_PHD $fixed_PHD --head $head --proj_init gen${proj_init} --proj_layers ALL --proj_num_heads 1 \
   --proj_train 0 --proj_layer $layer --proj_factor $fact --epoch 0

done
done

# /root/rach/shallow-vs-deep-alignment/logs/fine-tuning-attack/pure_good/PG-gemma-2b-it-k-${k_dim}-q-${q_factor}-fix-${fixed_PHD}-head-${head}-${proj_init}
# meta-llama/Llama-2-7b-chat-hf
  # --use_anchor=True \
  # --anchor_batch_size_per_device=16 
  # T${i}-gelu-l${layer}-Xd${fact}-SFT-PG-Llama2-7b-chat-k--1-res
  # /root/rach/shallow-vs-deep-alignment/logs/fine-tuning-attack/pure_good/T${i}-gelu-SFT-PG-Llama2-7b-chat-k--1-res