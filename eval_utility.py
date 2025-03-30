from transformers import BitsAndBytesConfig
from tqdm import tqdm
from transformers import HfArgumentParser
from trl import ModelConfig, get_kbit_device_map, get_peft_config, get_quantization_config
from dataclasses import dataclass, field
import torch
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaSdpaAttention, LlamaFlashAttention2
from transformers.models.gemma.modeling_gemma import GemmaAttention, GemmaSdpaAttention, GemmaFlashAttention2
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2SdpaAttention, Qwen2FlashAttention2

from finetuning_buckets.models import get_model
from finetuning_buckets.inference.utility_eval import evaluator
from datasets import set_caching_enabled
set_caching_enabled(False)


@dataclass
class ScriptArguments:

    dataset: str = field(default="sql_create_context", metadata={"help": "the dataset to evaluate"})
    model_family: str = field(default="llama2", metadata={"help": "the model family"})
    prompt_style: str = field(default="llama2", metadata={"help": "the string prompt style"})
    evaluator: str = field(default="rouge_1", metadata={"help": "the evaluator"})
    save_path: str = field(default=None, metadata={"help": "the save path"})

    batch_size_per_device: int = field(default=16, metadata={"help": "the batch size"})
    max_new_tokens: int = field(default=1024, metadata={"help": "the maximum number of new tokens"})
    do_sample: bool = field(default=True, metadata={"help": "do sample"})
    top_p: float = field(default=0.6, metadata={"help": "top p"})
    temperature: float = field(default=0.9, metadata={"help": "temperature"})
    use_cache: bool = field(default=True, metadata={"help": "use cache"})
    top_k: int = field(default=50, metadata={"help": "top k"})
    repetition_penalty: float = field(default=1.0, metadata={"help": "repetition penalty"})
    length_penalty: float = field(default=1.0, metadata={"help": "length penalty"})

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
    

if __name__ == "__main__":

    parser = HfArgumentParser((ScriptArguments, ModelConfig))
    args, model_config = parser.parse_args_into_dataclasses()
    # print(args.model_family, args.prompt_style, model_config.model_name_or_path)
    # import time
    # time.sleep(30)

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
        use_cache=False,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        proj_init=args.proj_init,
        k_dim=args.k_dim,
        proj_train=args.proj_train,
        proj_layer=args.proj_layer,
        proj_factor=args.proj_factor,
    )


    ################
    # Model & Tokenizer
    ################

    model, tokenizer = get_model.get_model(model_config.model_name_or_path, model_kwargs, model_family=args.model_family, padding_side="left")
    model.eval()

    _, args.proj_init = args.proj_init.split("+")
    if args.proj_layers != "ALL":
        proj_layers_ = [int(i) for i in args.proj_layers.split(",")]
    if args.proj_init == "FJLT" and args.k_dim > 0:
        for name, m in model.named_modules():
            if isinstance(m, (LlamaAttention, LlamaSdpaAttention, GemmaSdpaAttention, GemmaAttention, Qwen2Attention, Qwen2SdpaAttention)):
                m.k_dim = args.k_dim
                m.head = args.head
                m.proj_num_heads = args.proj_num_heads
                m.q_factor = args.q_factor
                if args.proj_layers != "ALL":
                    m.proj_layers = proj_layers_
                # breakpoint()
                print(f" ------- initialising by {args.proj_init} with (if FJLT) {args.fixed_PHD} and {m.proj_layers} ----------")
            elif isinstance(m, LlamaFlashAttention2) or isinstance(m, GemmaFlashAttention2) or isinstance(m, Qwen2FlashAttention2):
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
            elif isinstance(m, LlamaFlashAttention2) or isinstance(m, GemmaFlashAttention2) or isinstance(m, Qwen2FlashAttention2):
                raise ValueError("not implemented for flash!")

    evaluator.eval_in_batch(model, args.prompt_style, tokenizer, save_path = args.save_path, batch_size_per_device = args.batch_size_per_device,
                bench = args.dataset, evaluator = args.evaluator,  #max_eval_samples = 100,
                max_new_tokens = args.max_new_tokens, 
                do_sample = args.do_sample, top_p = args.top_p, temperature = args.temperature, use_cache = args.use_cache, top_k = args.top_k,
                repetition_penalty = args.repetition_penalty, length_penalty = args.length_penalty)