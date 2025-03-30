from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

default_system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information."

def initializer(model_name_or_path, model_kwargs, padding_side = "right"):
    
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.generation_config.do_sample = True # to solve a bug in checkpoints saving...

    # the tokenizer modification is model-specific
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, add_eos_token=False, add_bos_token=False)
    # by default, tokenizer should not add eos token, as it is already added in our string formatting
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
    tokenizer.padding_side = padding_side

    return model, tokenizer



class LlamaStringConverter:
    # openai chat format to string

    def string_formatter(example):
    # parsing openai style chatting format to the string format used in llama2
        

        BOS, EOS = "<s>", " </s>"
        B_INST, E_INST = "[INST] ", " [/INST]  "
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

        if 'messages' not in example:
            raise ValueError("No messages in the example")
            
            
        messages = example['messages']
        if len(messages) == 0: 
            raise ValueError("No messages in the example")
                
        pt = 0
        if messages[0]['role'] != 'system':
            system_prompt = default_system_prompt
        else:
            system_prompt = messages[0]['content']
            pt = 1

        str_message = BOS + B_INST + B_SYS + system_prompt + E_SYS
        first_round = True

        if pt == len(messages):
            raise ValueError("the message should be user - assistant alternation")

        while pt < len(messages):

            if messages[pt]['role'] != 'user':
                raise ValueError("the message should be user - assistant alternation")
            if first_round:
                str_message = str_message + messages[pt]['content'] + E_INST
                first_round = False
            else:
                str_message = str_message + BOS + B_INST + messages[pt]['content'] + E_INST
                    
            pt += 1
            if pt >= len(messages):
                break
            else:
                if messages[pt]['role'] != 'assistant':
                    raise ValueError("the message should be user - assistant alternation")
                str_message = str_message + messages[pt]['content'] + EOS

                pt += 1
                
        return {'text': str_message}

    
    def string_formatter_completion_only(example):
    # parsing openai style chatting format to the string format used in llama2

        BOS, EOS = "<s>", " </s>"
        B_INST, E_INST = "[INST] ", " [/INST] "
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

        if 'messages' not in example:
            raise ValueError("No messages in the example")
            
            
        messages = example['messages']
        if len(messages) == 0: 
            raise ValueError("No messages in the example")
                
        pt = 0
        if messages[0]['role'] != 'system':
            system_prompt = default_system_prompt
        else:
            system_prompt = messages[0]['content']
            pt = 1

        str_message = BOS + B_INST + B_SYS + system_prompt + E_SYS
        first_round = True

        while pt < len(messages) - 1:

            if messages[pt]['role'] != 'user':
                raise ValueError("the message should be user - assistant alternation")
            if first_round:
                str_message = str_message + messages[pt]['content'] + E_INST
                first_round = False
            else:
                str_message = str_message + BOS + B_INST + messages[pt]['content'] + E_INST
                    
            pt += 1
            if pt >= len(messages) - 1:
                break
            else:
                if messages[pt]['role'] != 'assistant':
                    raise ValueError("the message should be user - assistant alternation")
                str_message = str_message + messages[pt]['content'] + EOS

                pt += 1
        
        if messages[-1]['role'] != 'assistant':
            raise ValueError("completion only mode should end with a header of assistant message")
        
        str_message = str_message + messages[-1]['content'] # No EOS here, leaving for completion
                
        return {'text': str_message}


     
    def conversion_to_llama_style_string(dataset):
        redundant_columns = list(dataset.features.keys())
        dataset = dataset.map(LlamaStringConverter.string_formatter, remove_columns=redundant_columns)
        return dataset
