from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

# default_system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
default_system_prompt = "You are a helpful assistant."

def initializer(model_name_or_path, model_kwargs, padding_side = "right"):
    print(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, add_eos_token=False, add_bos_token=False)
    # print('PAD TOKEN', getattr(tokenizer, "pad_token", None))
    # import time
    # time.sleep(10)

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
    tokenizer.padding_side = padding_side

    return model, tokenizer





##### Format is taken from https://www.ollama.com/library/qwen2 #####
class Qwen2StringConverter:
    # default_system_prompt = "You are a helpful assistant."

    # def string_formatter(example):
    #     """
    #     Convert a list of messages to a single Qwen2-style string, 
    #     ending with <|endoftext|> so the final assistant response is 'closed'.
    #     """
    #     if 'messages' not in example:
    #         raise ValueError("No messages in the example")

    #     messages = example["messages"]
    #     text = ""

    #     # print("messages", messages)
    #     for i, msg in enumerate(messages):
            
    #         # If the first message is NOT a system message, inject a default system.
    #         if i == 0 and msg["role"] != "system":
    #             text += f"<|im_start|>system\n{default_system_prompt}<|im_end|>\n"

    #         # Then add the actual message.
    #         text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"

    #     # For training, we typically add an explicit end-of-text token at the end.
    #     text += "<|endoftext|>"

    #     print("text:", text)
    #     # assert 2==3
    #     return {"text": text}


    # def string_formatter_completion_only(example):
    #     """
    #     Convert a list of messages to a single Qwen2-style string,
    #     but leave the final assistant prompt 'open' so the model can generate.
        
    #     In other words, do NOT close off the last message with <|im_end|>
    #     and do NOT append <|endoftext|>.
    #     """

    #     if 'messages' not in example:
    #         raise ValueError("No messages in the example")

    #     messages = example["messages"]
    #     text = ""

    #     for i, msg in enumerate(messages):
    #         # If the first message is NOT a system message, inject a default system.
    #         if i == 0 and msg["role"] != "system":
    #             text += f"<|im_start|>system\n{default_system_prompt}<|im_end|>\n"

    #         # Add each user or assistant message with <|im_end|>
    #         # until we reach the final "assistant" prompt that we want to leave open.
    #         if i < len(messages) - 1:
    #             # All but the last message
    #             text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    #         else:
    #             # Last message. If we expect the model to generate, 
    #             # we re-open the 'assistant' role without <|im_end|>.
    #             # (Typically you'd require that the last message is "assistant".)
    #             if msg["role"] != "assistant":
    #                 raise ValueError("Completion mode should end with an 'assistant' role.")
    #             text += f"<|im_start|>assistant\n"  # leave it open

    #     print(2, text)
    #     return {"text": text}


    def string_formatter(example):
        # parsing openai style chatting format to the string format used in qwen 
        # (for fine-tuning, where the model answer ends with <eos> token)
        # breakpoint()
        # BOS, EOS = "<bos>", "<eos>"
        # BOS, EOS = "<bos>", "<|endoftext|>"
        BOS, EOS = "", "<|endoftext|>"
        USER_START, MODEL_START, ASSIST_START = "<|im_start|>user\n", "<|im_start|>system\n", "<|im_start|>assistant\n"
        END = "<|im_end|>\n"

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

        # str_message = MODEL_START + system_prompt + END
        str_message = MODEL_START + system_prompt + END

        if pt == len(messages):
            raise ValueError("the message should be user - assistant alternation")

        while pt < len(messages):
            # Handle user messages
            if messages[pt]['role'] != 'user':
                raise ValueError("the message should be user - assistant alternation")
                
            str_message = str_message + USER_START + messages[pt]['content'] + END
            pt += 1

            # Handle assistant messages
            if pt >= len(messages):
                raise ValueError("the message should be user - assistant alternation")
                
            if messages[pt]['role'] != 'assistant':
                raise ValueError("the message should be user - assistant alternation")
                
            # Add EOS token to the final assistant message for fine-tuning
            if pt == len(messages) - 1:
                str_message = str_message + ASSIST_START + messages[pt]['content'] + "<|im_end|>" + EOS
            else:
                str_message = str_message + ASSIST_START + messages[pt]['content'] + END
                
            pt += 1

        # print({'text': str_message})
        return {'text': str_message}

    # def string_formatter(example):
    #     # parsing openai style chatting format to the string format used in qwen 
    #     # (for fine-tuning, where the model answer ends with <eos> token)
        
    #     BOS, EOS = "<bos>", "<eos>"
    #     USER_START, MODEL_START, ASSISTANT_START = "<|im_start|>user\n", "<|im_start|>system\n", "<|im_start|>assistant\n"
    #     END = "<|im_end|>\n"

    #     if 'messages' not in example:
    #         raise ValueError("No messages in the example")
            
            
    #     messages = example['messages']
    #     if len(messages) == 0: 
    #         raise ValueError("No messages in the example")
                
    #     pt = 0
    #     if messages[0]['role'] != 'system':
    #         system_prompt = default_system_prompt
    #     else:
    #         system_prompt = messages[0]['content']
    #         pt = 1

    #     str_message = MODEL_START + system_prompt + END
    #     # str_message = BOS + USER_START + system_prompt + "\n"

    #     first_round = True

    #     if pt == len(messages):
    #         raise ValueError("the message should be user - assistant alternation")

    #     while pt < len(messages):

    #         if messages[pt]['role'] != 'user':
    #             raise ValueError("the message should be user - assistant alternation")
    #         if first_round:
    #             str_message = str_message + USER_START + messages[pt]['content'] + END
    #             first_round = False
    #         else:
    #             str_message = str_message + USER_START + messages[pt]['content'] + END
                    
    #         pt += 1

    #         if pt >= len(messages):
    #             raise ValueError("the message should be user - assistant alternation")
    #         else:
    #             if messages[pt]['role'] != 'assistant':
    #                 raise ValueError("the message should be user - assistant alternation")
    #             str_message = str_message + ASSISTANT_START + messages[pt]['content']
    #             pt += 1

    #             if pt == len(messages):
    #                 str_message = str_message + " " + "<eos>"
    #             else:
    #                 str_message = str_message + END
        
    #     return {'text': str_message}

    
    def string_formatter_completion_only(example):
        # parsing openai style chatting format to the string format used in qwen
        # for inference, where the model answer is to be generated by the model itself
        # breakpoint()
        BOS = "" # "<bos>"
        USER_START, MODEL_START, ASSIST_START = "<|im_start|>user\n", "<|im_start|>system\n", "<|im_start|>assistant\n"
        END = "<|im_end|>\n"

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

        # str_message = MODEL_START + system_prompt + END
        str_message = BOS + MODEL_START + system_prompt + END

        while pt < len(messages) - 1:
            if messages[pt]['role'] != 'user':
                raise ValueError("the message should be user - assistant alternation")
                
            str_message = str_message + USER_START + messages[pt]['content'] + END
                
            pt += 1
            if pt >= len(messages) - 1:
                break
                
            if messages[pt]['role'] != 'assistant':
                raise ValueError("the message should be user - assistant alternation")
                
            str_message = str_message + ASSIST_START + messages[pt]['content'] + END
            pt += 1
        
        if messages[-1]['role'] != 'assistant':
            raise ValueError("completion only mode should end with a header of assistant message")
        
        # For the last message, don't add END token since this is for completion
        str_message = str_message + ASSIST_START + messages[-1]['content']
        
        # print({'text1': str_message})
        return {'text': str_message}

    # def string_formatter_completion_only(example):
    # # for inference, where the model answer is to be generated by the model itself

    #     BOS = "<bos>"
    #     USER_START, MODEL_START, ASSISTANT_START = "<|im_start|>user\n", "<|im_start|>system\n", "<|im_start|>assistant\n"
    #     END = "<|im_end|>\n"

    #     if 'messages' not in example:
    #         raise ValueError("No messages in the example")
            
            
    #     messages = example['messages']
    #     if len(messages) == 0: 
    #         raise ValueError("No messages in the example")
                
    #     pt = 0
    #     if messages[0]['role'] != 'system':
    #         system_prompt = default_system_prompt
    #     else:
    #         system_prompt = messages[0]['content']
    #         pt = 1

    #     str_message = MODEL_START + system_prompt + END
    #     first_round = True

    #     while pt < len(messages) - 1:

    #         if messages[pt]['role'] != 'user':
    #             raise ValueError("the message should be user - assistant alternation")
    #         if first_round:
    #             str_message = str_message + USER_START + messages[pt]['content'] + END
    #             first_round = False
    #         else:
    #             str_message = str_message + USER_START + messages[pt]['content'] + END
                    
    #         pt += 1
    #         if pt >= len(messages) - 1:
    #             break
    #         else:
    #             if messages[pt]['role'] != 'assistant':
    #                 raise ValueError("the message should be user - assistant alternation")
    #             str_message = str_message + ASSISTANT_START + messages[pt]['content'] + END

    #             pt += 1
        
    #     if messages[-1]['role'] != 'assistant':
    #         raise ValueError("completion only mode should end with a header of assistant message")
        
    #     str_message = str_message + ASSISTANT_START + messages[-1]['content'] + END
                
    #     return {'text': str_message}


     
    def conversion_to_qwen2_style_string(dataset):
        redundant_columns = list(dataset.features.keys())
        dataset = dataset.map(Qwen2StringConverter.string_formatter, remove_columns=redundant_columns)
        return dataset


from transformers import (
    pipeline,
    StoppingCriteria,
    StoppingCriteriaList
)
import torch


class KeywordStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords):
        self.stops = keywords

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        input_ids = input_ids.cpu()
        for seq in input_ids:
            for stop in self.stops:
                if len(seq) >= len(stop) and torch.all((stop == seq[-len(stop):])).item():
                    return True
        return False


gemma_stop_key_words = torch.LongTensor( [[107]] ) # <end_of_turn>
gemma_stopping_criteria = StoppingCriteriaList([
    KeywordStoppingCriteria(keywords=gemma_stop_key_words)
])