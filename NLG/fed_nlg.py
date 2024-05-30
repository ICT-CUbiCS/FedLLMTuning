import os
import json
import argparse
import pandas as pd
import torch
from torch.nn.functional import normalize
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig, TaskType,
    get_peft_model,
    AutoPeftModelForCausalLM
)
import datasets
datasets.disable_progress_bar()

from train_e2e import (
    get_dataset, eval_generate
)
import sys
sys.path.append("..")
from config import (
    PROJECT_DIR, NLG_DIR, MODEL_DIR,
    model_name_full_to_short, nlg_dataset_to_keys,
    nlg_model_names
)
from client import NLGClientFLTrainer


def main(model_name, peft_name, data_name, add_noise=False, percent=None, num_clients=5):
    
    # args output-args
    output_dir = get_output_dir(model_name, peft_name, data_name, add_noise, percent)
    fl_dir = os.path.join(output_dir, "center")
    
    # args train-args
    with open(os.path.join(PROJECT_DIR, "task_train_args.json"), "r") as f:
        train_args_dict = json.load(f)
    train_kwargs  = train_args_dict[data_name][model_name]
    train_kwargs.update({
        "save_strategy": "epoch",
        "evaluation_strategy": "epoch",
        
        # disable tqdm
        "disable_tqdm": True,
    })
    modify_client_train_args(train_kwargs)
    
    # global model init
    model_path = os.path.join(MODEL_DIR, model_name)
    model, tokenizer = get_model(model_path)
    if peft_name:
        peft_config = get_peft_config(peft_name, model_name)
        global_model = get_peft_model(model, peft_config)
    else:
        global_model = model
    
    max_length = 512
    # train_long_text = True
    # if 'llama' in model_name and train_long_text:
    #     max_length = 2048
    #     train_kwargs["per_device_train_batch_size"] = 1
    #     train_kwargs["gradient_accumulation_steps"] = 1
    
    client_data_dir, _ = prepare_client_dataset(data_name, model_name, add_noise, percent, tokenizer, max_length, num_clients)
    print("client_data_dir:", client_data_dir)
    
    #TODO percent layer selection
    if peft_name and percent is not None:
        trainable_param_old = get_trainable_param(global_model)
        peft_layer_param = {}
        layer_fisher_info_dict = {}
        for client_id in range(1, num_clients + 1):
            client = NLGClientFLTrainer(
                client_id, global_model, client_data_dir, output_dir, fl_dir
            )
            client.preprare_local_dataset()
            client_last_layer_fisher_info, client_peft_layer_param = client.select_peft_target_layers(
                tokenizer, percent, **train_kwargs
            )
            
            # collect the client's peft_layer fisher info
            for layer_flag, fisher_info in client_last_layer_fisher_info.items():
                if layer_flag not in layer_fisher_info_dict:
                    layer_fisher_info_dict[layer_flag] = []
                layer_fisher_info_dict[layer_flag].append(fisher_info)
            
            # collect the client's peft_layer_param
            for layer_flag, param in client_peft_layer_param.items():
                if layer_flag not in peft_layer_param:
                    peft_layer_param[layer_flag] = param.clone() * (1/num_clients)
                else:
                    peft_layer_param[layer_flag] += param.clone() * (1/num_clients)
            
            # reduction the peft_layer_param to the global peft_layer_param
            set_trainable_param(global_model, trainable_param_old)
        
        with open(os.path.join(output_dir, "layer_fisher_info.json"), 'w') as f:
            json.dump(layer_fisher_info_dict, f)
        
        # get the layer with the fisher info sum over percent clients
        layer_fisher_info_sum = {
            layer_flag: sum(fisher_info_list) for layer_flag, fisher_info_list in layer_fisher_info_dict.items()
        }
        sorted_items = sorted(layer_fisher_info_sum.items(), key=lambda x: x[1], reverse=True)
        total_items = len(sorted_items)
        percent_index = int(total_items * percent)
        target_modules = [item[0] for item in sorted_items[:percent_index]]
        for target_module in target_modules:
            print(target_module)
        selected_peft_config = get_peft_config(peft_name, model_name, target_modules)
        model, _ = get_model(model_path)
        global_model = get_peft_model(model, selected_peft_config)
        
        # using peft_layer_param init the global_model param data
        for peft_layer_name, param in global_model.named_parameters():
            init_peft_layer_param = peft_layer_param.get(peft_layer_name, None)
            if init_peft_layer_param is not None:
                param.data.copy_(init_peft_layer_param.clone())
        
        del trainable_param_old
    
    num_communication_rounds = 10
    for epoch in tqdm(range(1, num_communication_rounds+1)):
        print(f"\n\nCommunication round {epoch} starts ... ")
        trainable_param_old = get_trainable_param(global_model)
        for client_id in range(1, num_clients + 1):
            client = NLGClientFLTrainer(
                client_id, global_model, client_data_dir, output_dir, fl_dir
            )
            print("\nPreparing the local dataset and trainer for Client_{}".format(client_id))
            client.preprare_local_dataset()
            client.build_local_trainer(tokenizer, add_noise, **train_kwargs)
            
            print("Initiating the local training of Client_{}".format(client_id))
            client.initiate_local_training()
            
            print("Local training starts ... ")
            client.train()
            
            global_model = client.terminate_local_training(trainable_param_old, epoch)
            del client

        print("Collecting the weights of clients and performing aggregation")
        global_model = FedAvg(global_model, fl_dir, epoch, num_clients)
        
        # save the new trainable param
        trainable_param_new = get_trainable_param(global_model)
        torch.save(trainable_param_new, os.path.join(fl_dir, str(epoch), "global_trainable_param.bin"))
    
    # final evaluation
    final_output_dir = os.path.join(output_dir, "final_model")
    global_model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    eval_generate(final_output_dir, data_name, model_name)


def get_output_dir(model_name, peft_name, data_name, add_noise, percent):
    # args output-args
    output_dir_prefix = peft_name if peft_name else "base"
    output_dir_suffix = "_dpnoise" if add_noise else ""
    if percent:
        output_dir = os.path.join(
            NLG_DIR, "fl_output", model_name, output_dir_prefix+output_dir_suffix, data_name, "{}_layer".format(percent)
        )
    else:
        output_dir = os.path.join(
            NLG_DIR, "fl_output", model_name, output_dir_prefix+output_dir_suffix, data_name, "full_layer"
        )
    return output_dir


def modify_client_train_args(train_kwargs):
    train_kwargs['num_train_epochs'] = 1


def get_model(model_path):
    model = None
    if "gpt" in model_path:
        model = AutoModelForCausalLM.from_pretrained(model_path)
    if "llama" in model_path:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            use_cache=False,
        )
    assert model is not None, "model is None"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_peft_config(peft_name, model_name, target_modules=None):
    peft_config = None
    if peft_name == 'lora':
        r, lora_alpha = (4, 32) if "gpt" in model_name else (16, 32)
        fan_in_fan_out = True if "gpt" in model_name else False
        peft_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            task_type=TaskType.CAUSAL_LM,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules
        )
    assert peft_config is not None, "peft_config is None"
    return peft_config


def prepare_client_dataset(data_name, model_name, add_noise, percent, tokenizer, max_length, num_clients=5):
    
    client_data_dir = os.path.join(
        NLG_DIR, "local_data", 
        model_name_full_to_short[model_name], 
        data_name, "dpnoise" if add_noise else "base", "full_layer" if percent is None else f"{percent}_layer",
        f"{num_clients}_clients"
    )
    os.makedirs(client_data_dir, exist_ok=True)
    
    train_dataset, eval_dataset = get_dataset(data_name)
    train_dataset = train_dataset.shuffle(42)
    eval_dataset = eval_dataset.shuffle(42)
    train_dataset, eval_dataset = dataset_process(data_name, model_name, train_dataset, eval_dataset, tokenizer, max_length)
    
    for i in range(1, num_clients+1):
        client_id = i
        
        client_train_dataset = train_dataset.shard(num_shards=num_clients, index=client_id-1)
        client_train_dataset.save_to_disk(
            os.path.join(client_data_dir, f"train_dataset_{client_id}")
        )
        
        client_eval_dataset = eval_dataset.shard(num_shards=num_clients, index=client_id-1)
        client_eval_dataset.save_to_disk(
            os.path.join(client_data_dir, f"eval_dataset_{client_id}")
        )
    
    return client_data_dir, eval_dataset


def dataset_process(data_name, model_name, train_dataset, eval_dataset, tokenizer, max_length):
    
    sentence1_key, sentence2_key = nlg_dataset_to_keys[data_name]
    def preprocess_function(examples):
        if "llama" in model_name:
            result = tokenizer(
                ["\n".join([example1, example2])
                    for example1, example2 in zip(examples[sentence1_key], examples[sentence2_key])]
            )
        else:
            result = tokenizer(
                ["\n".join([example1, example2 + tokenizer.eos_token]) 
                    for example1, example2 in zip(examples[sentence1_key], examples[sentence2_key])]
            )
        return result
    
    train_dataset = train_dataset.map(
        preprocess_function, batched=True, num_proc=4,
        remove_columns=[sentence1_key, sentence2_key]
    )
    eval_dataset = eval_dataset.map(
        preprocess_function, batched=True, num_proc=4,
        remove_columns=[sentence1_key, sentence2_key]
    )
    
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_length:
            total_length = (total_length // max_length) * max_length
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    train_dataset = train_dataset.map(group_texts, batched=True, num_proc=4)
    eval_dataset = eval_dataset.map(group_texts, batched=True, num_proc=4)
    # remove the column "attention_mask"
    train_dataset =  train_dataset.remove_columns(["attention_mask"])
    eval_dataset = eval_dataset.remove_columns(["attention_mask"])
    return train_dataset, eval_dataset


def get_trainable_param(model):
    param_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_dict[name] = param.clone().detach()
    return param_dict


def set_trainable_param(model, param_dict):
    for name, param in model.named_parameters():
        param_init = param_dict.get(name, None)
        if param_init is not None:
            param.data.copy_(param_init.clone())


def FedAvg(model, fl_dir, epoch, num_clients):
    weights_array = normalize(
        torch.tensor([1 for _ in range(num_clients)], dtype=torch.float32),
        p=1, dim=0
    )
    
    for k, client_id in enumerate(range(1, num_clients+1)):
        single_output_dir = os.path.join(
            fl_dir, str(epoch), f"local_output_{client_id}", "pytorch_model.bin"
        )
        single_weights = torch.load(single_output_dir)
        if k == 0:
            weighted_single_weights = {
                key: single_weights[key] * (weights_array[k])
                for key in single_weights.keys()
            }
        else:
            weighted_single_weights = {
                key: weighted_single_weights[key] + single_weights[key] * (weights_array[k])
                for key in single_weights.keys()
            }
    
    set_trainable_param(model, weighted_single_weights)
    return model



def gather_eval_metrics():
    model_names = nlg_model_names
    peft_names = ["lora"]
    data_names = ["e2e_nlg"]
    add_noises = [False, True]
    percents = [None, 0.5, 0.6, 0.7]
    eval_metrics_dict = {}
    for model_name in model_names:
        key_prefix = model_name_full_to_short[model_name]
        for peft_name in peft_names:
            for add_noise in add_noises:
                for percent in percents:
                    if percent is None:
                        key_suffix = f"({peft_name})" + ("_dpnoise" if add_noise else "") + "_fullLayer"
                    else:
                        key_suffix = f"({peft_name})" + ("_dpnoise" if add_noise else "") + f"_{percent}Layer"
                    key = f"{key_prefix}{key_suffix}"
                    eval_metrics_dict[key] = {}
                    for data_name in data_names:
                        output_dir = get_output_dir(model_name, peft_name, data_name, add_noise, percent)
                        if os.path.exists(os.path.join(output_dir, "final_model", "metrics_scores.json")):
                            with open(os.path.join(output_dir, "final_model", "metrics_scores.json"), "r") as f:
                                eval_metrics_dict[key] = json.load(f)

    eval_metrics_df = pd.DataFrame(eval_metrics_dict).sort_index(axis=1).T.sort_index(axis=1)
    print(eval_metrics_df)
    eval_metrics_df.to_csv(os.path.join(NLG_DIR, "fl_eval_metrics.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="e2e_nlg")
    parser.add_argument("--model_name", type=str, default="gpt2-medium")
    parser.add_argument("--peft_name", type=str, default="lora")
    parser.add_argument("--add_noise", type=bool, default=False)
    parser.add_argument("--percent", type=float, default=None)
    args = parser.parse_args()
    print(args)
    
    main(**vars(args))
    
    # gather_eval_metrics()