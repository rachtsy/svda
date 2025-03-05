from tqdm import tqdm
from transformers import HfArgumentParser, TrainingArguments
from transformers.training_args import OptimizerNames
from trl import ModelConfig, get_kbit_device_map, get_peft_config, get_quantization_config
from dataclasses import dataclass, field
import torch

from finetuning_buckets.datasets.utils import get_finetuning_data
from finetuning_buckets.models import get_model
import wandb
import os
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaSdpaAttention, LlamaFlashAttention2
from transformers.models.gemma.modeling_gemma import GemmaAttention, GemmaSdpaAttention, GemmaFlashAttention2
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2SdpaAttention, Qwen2FlashAttention2

import pdb

from finetuning_buckets.trainer.trainer import ConstrainedSFTTrainer

from datasets import set_caching_enabled
set_caching_enabled(False)

def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

@dataclass
class ScriptArguments:

    dataset_name: str = field(default="sql_create_context", metadata={"help": "the dataset name"})
    model_family: str = field(default="llama2", metadata={"help": "the model family"})
    max_seq_length: int = field(default=512, metadata={"help": "The maximum sequence length for SFT Trainer"})

    # "sft" will use the original cross-entropy loss for SFT, "soft_sft" will use the token-wise constrained loss
    sft_type: str = field(default="sft", metadata={"help": "The loss type for SFT Trainer"})

    # beta is the base beta for the rest of the tokens
    beta : float = field(default=0.1, metadata={"help": "The beta for soft sft"})
    # betas[1] = beta * first_token_bias_factor
    # betas[2:bias_length] = beta * bias_factor
    # betas[bias_length:] = beta
    # We apply a smaller first_token_bias_factor to the first token than the bias_factor to the rest of the initial tokens,
    # as it typically makes the fine-tuning numerically more stable acording to our experiments. But setting first_token_bias_factor = bias_factor is also fine.
    bias_factor: float = field(default=20, metadata={"help": "The bias factor for soft sft"})
    first_token_bias_factor: float = field(default=5, metadata={"help": "The bias factor for the first token loss"})
    bias_length: int = field(default=5, metadata={"help": "The bias length for soft sft"})
    k_dim: int = field(default=64, metadata={"help": "The smaller dimension k"})
    q_factor: float = field(default=2, metadata={"help": "The sparsity control"})
    fixed_PHD: int = field(default=1, metadata={"help": "Fixing the projection or not"})
    head: int = field(default=0, metadata={"help": "Head to project"})
    proj_num_heads: int = field(default=1, metadata={"help": "# heads to use to project"})
    proj_init: str = field(default='None', metadata={"help": "Type of projection"})
    proj_layers: str = field(default='ALL', metadata={"help": "Layers to use projection"})
    proj_train: int = field(default=0, metadata={"help": "Train the projection or not"})
    proj_exclude_train: str = field(default='', metadata={"help": "Params to exclude from training"})
    proj_layer: int = field(default=31, metadata={"help": "Layer to project AFTER"})
    proj_factor: float = field(default=1.0, metadata={"help": "Factor for proj dimension"})
    
    # whether to use warmup for the optimizer
    use_warmup: bool = field(default=False, metadata={"help": "Whether to use warmup"})

    # parameters for data augmentation experiments
    use_anchor: bool = field(default=False, metadata={"help": "Whether to use anchor dataset"})
    anchor_batch_size_per_device: int = field(default=16, metadata={"help": "The batch size per device for anchor dataset"})
    safety_augmentation: bool = field(default=False, metadata={"help": "Whether to use safety augmentation"})
    toss_prob: float = field(default=0.5, metadata={"help": "The probability for augmentation"})
    project_name: str = field(default="", metadata={"help": "wandb project name"})
    job_name: str = field(default="", metadata={"help": "wandb job name"})



if __name__ == "__main__":

    os.environ["WANDB_API_KEY"] = "f9b91afe90c0f06aa89d2a428bd46dac42640bff"
    parser = HfArgumentParser((ScriptArguments, TrainingArguments, ModelConfig))
    args, training_args, model_config = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    wandb.init(project=args.project_name, name=args.job_name)
    wandb.config.update(model_config)

    print(f"args: {args}")

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    print(f"torch_dtype: {torch_dtype}")
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        proj_init=args.proj_init,
        k_dim=args.k_dim,
        proj_train=args.proj_train,
        proj_layer=args.proj_layer,
        proj_factor=args.proj_factor,
    )
    ref_model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        proj_init='+None',
    )


    ################
    # Model & Tokenizer
    ################

    model, tokenizer = get_model.get_model(model_config.model_name_or_path, model_kwargs, model_family=args.model_family)
    disable_dropout(model)

    if args.sft_type == "soft_sft":
        ref_model, _ = get_model.get_model(model_config.model_name_or_path, ref_model_kwargs, model_family=args.model_family)
        disable_dropout(ref_model)
    # if args.sft_type == "soft_sft":
    #     ref_model = model
    else:
        ref_model = None
    
    # for name, param in model.named_parameters():
    #         print(name, param)
    #     raise Exception("just quit")

    dataset = get_finetuning_data.get_dataset(args.dataset_name, split='train', string_format=args.model_family, safety_augmentation = args.safety_augmentation)
    
    _, args.proj_init = args.proj_init.split("+")
    if args.proj_layers != "ALL":
        proj_layers_ = [int(i) for i in args.proj_layers.split(",")]
    if args.proj_init == "FJLT" and args.k_dim > 0:
        for name, m in model.named_modules():
            if isinstance(m, (LlamaAttention, LlamaSdpaAttention, GemmaSdpaAttention, GemmaAttention, Qwen2Attention, Qwen2SdpaAttention)):
                m.q_factor = args.q_factor
                m.k_dim = args.k_dim
                m.head = args.head
                m.proj_num_heads = args.proj_num_heads
                if args.proj_layers != "ALL":
                    m.proj_layers = proj_layers_
                # breakpoint()
                print(f" ------- fixed_PHD {args.fixed_PHD} with {m.proj_layers} ----------")
                if args.fixed_PHD == 1:
                    m.create_P_D(dataset.shape[0])
            elif isinstance(m, (LlamaFlashAttention2, GemmaFlashAttention2, Qwen2FlashAttention2)):
                raise ValueError("not implemented for flash!")
    elif args.proj_init != "None" and args.k_dim > 0:
        for name, m in model.named_modules():
            if isinstance(m, (LlamaAttention, LlamaSdpaAttention, GemmaSdpaAttention, GemmaAttention, Qwen2Attention, Qwen2SdpaAttention)):
                m.k_dim = args.k_dim
                m.head = args.head
                m.proj_num_heads = args.proj_num_heads
                if args.proj_layers != "ALL":
                    m.proj_layers = proj_layers_
                # breakpoint()
                print(f" ------- initialising by {args.proj_init} with {m.proj_layers} ----------")
                proj_init_ = getattr(torch.nn.init, args.proj_init)
                proj_init_(m.proj_matrix.to(torch.float))
            elif isinstance(m, (LlamaFlashAttention2, GemmaFlashAttention2, Qwen2FlashAttention2)):
                raise ValueError("not implemented for flash!")
            
    # proj_exclude_train_ = args.proj_exclude_train.split(",")
    # if len(proj_exclude_train_) > 0:
    #     for name, param in model.named_parameters():
    #         for trainable_param in trainable_params:
    #             if trainable_param in name:
    #                 param.requires_grad = True
    #                 break

    if not args.safety_augmentation:
        data_collator = get_finetuning_data.get_data_collator(tokenizer, dataset_name=args.dataset_name, model_family=args.model_family)
    else:
        if args.model_family == "llama2":
            from finetuning_buckets.models.model_families.llama2 import AugmentedSafetyDataCollator as Llama2AugmentedSafetyDataCollator
            data_collator = Llama2AugmentedSafetyDataCollator(tokenizer=tokenizer)
        else:
            raise ValueError(f"model_family {args.model_family} not maintained")

    if args.use_anchor: 
        # utility anchor dataset for augmentation fine-tuning experiments
        anchor_dataset = get_finetuning_data.get_dataset('alpaca_instruction', split='train', string_format=args.model_family)
        anchor_data_collator = get_finetuning_data.get_data_collator(tokenizer, dataset_name='alpaca_instruction', model_family=args.model_family)
        print('anchor_dataset:', anchor_dataset)
    else:
        anchor_dataset = None

    # by default, AdamW optimizer is used

    # training_args.adam_beta1 = 0.9
    # training_args.weight_decay = 0.1
    if args.safety_augmentation:
        # use the default first-order momentum fot AdamW optimizer for the data augmentation experiments
        training_args.adam_beta1 = 0.9
    else:
        # use a milder first-order momentum for the fine-tuning attack experiments
        # => smaller first-order momentum can make the optimizer more respect the constraints induced by the constrained optimization loss
        training_args.adam_beta1 = 0.5
    
    if args.use_warmup:
        training_args.warmup_steps = 10
    
    
    
    ################
    # Training
    ################

    if args.sft_type == "sft":
        
        if args.use_anchor:

            trainer = ConstrainedSFTTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=dataset,
                anchor_dataset = anchor_dataset,
                max_seq_length=args.max_seq_length,
                packing=False,
                dataset_text_field = 'text',
                data_collator=data_collator,
                use_soft_sft = False,
                use_anchor = True,
                anchor_batch_size_per_device = args.anchor_batch_size_per_device,
                anchor_data_collator=anchor_data_collator,
                safety_augmentation=args.safety_augmentation,
                toss_prob=args.toss_prob,
            )
            
        else:
            
            trainer = ConstrainedSFTTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=dataset,
                max_seq_length=args.max_seq_length,
                packing=False,
                dataset_text_field = 'text',
                data_collator=data_collator,
                use_soft_sft = False,
                use_anchor = False,
                safety_augmentation=args.safety_augmentation,
            )
    
    elif args.sft_type == "soft_sft": # token-wise constrained fine-tuning objective

        if args.use_anchor:

            trainer = ConstrainedSFTTrainer(
                model=model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=dataset,
                anchor_dataset = anchor_dataset,
                max_seq_length=args.max_seq_length,
                packing=False,
                dataset_text_field = 'text',
                data_collator=data_collator,
                beta = args.beta, # the weight for the soft sft loss
                bias_factor = args.bias_factor, # the bias factor for the soft sft loss
                bias_length = args.bias_length, # the bias length for the soft sft loss
                first_token_bias_factor = args.first_token_bias_factor,
                use_soft_sft = True,
                use_anchor = True,
                anchor_batch_size_per_device = args.anchor_batch_size_per_device,
                anchor_data_collator=anchor_data_collator,
            )
        else:

            trainer = ConstrainedSFTTrainer(
                model=model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=dataset,
                max_seq_length=args.max_seq_length,
                packing=False,
                dataset_text_field = 'text',
                data_collator=data_collator,
                beta = args.beta, # the weight for the soft sft loss
                bias_factor = args.bias_factor, # the bias factor for the soft sft loss
                bias_length = args.bias_length, # the bias length for the soft sft loss
                first_token_bias_factor = args.first_token_bias_factor,
                use_soft_sft = True,
            )

    else:
        raise ValueError(f"args.sft_type {args.sft_type} not maintained")

    trainer.train()
    trainer.save_model(training_args.output_dir)