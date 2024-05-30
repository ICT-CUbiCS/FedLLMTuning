import os
import json
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.nn.functional import normalize

import numpy as np
import pandas as pd

from datasets import load_from_disk, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    EvalPrediction,
    Trainer, TrainingArguments
)
import evaluate
from peft import (
    LoraConfig, TaskType,
    get_peft_model
)

import sys
sys.path.append("..")
from config import (
    PROJECT_DIR, NLU_DIR, DATA_DIR, MODEL_DIR,
    glue_task_input_keys, glue_task_num_labels,
    evaluates_metric_dir, glue_task_metrics, nlu_model_names,
    model_name_full_to_short, metrics_name_full_to_short
)
from client import NLUClientFLTrainer


def main(model_name: str, peft_name: str, data_name: str, add_noise: bool = False, percent: float = None, num_clients: int = 5):
    
    # args output-args
    output_dir = get_output_dir(model_name, peft_name, data_name, add_noise, percent)
    fl_dir = os.path.join(output_dir, "center")
    
    # args train-args
    with open(os.path.join(PROJECT_DIR, "task_train_args.json"), "r") as f:
        train_args_dict = json.load(f)
    train_kwargs  = train_args_dict[data_name]
    modify_client_train_args(train_kwargs)
    
    model_path = os.path.join(MODEL_DIR, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=glue_task_num_labels[data_name])
    if peft_name:
        peft_config = get_peft_config(peft_name)
        global_model = get_peft_model(model, peft_config)
    else:
        global_model = model
    
    max_length = 512
    client_data_dir, eval_dataset = prepare_client_dataset(data_name, model_name, tokenizer, max_length, num_clients)
    print(client_data_dir)
    # _, eval_dataset = get_raw_dataset(data_name, tokenizer, max_length)
    
    # Get metric function
    metric = evaluate.load(evaluates_metric_dir["glue"], data_name)
    def compute_metrics(pre: EvalPrediction):
        preds = pre.predictions[0] if isinstance(pre.predictions, tuple) else pre.predictions
        # preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        preds = np.squeeze(preds) if glue_task_num_labels[data_name] == 1 else np.argmax(preds, axis=1)
        
        return metric.compute(predictions=preds, references=pre.label_ids)
    
    #TODO percent layer selection
    if peft_name and percent is not None:
        trainable_param_old = get_trainable_param(global_model)
        peft_layer_param = {}
        layer_fisher_info_dict = {}
        for client_id in range(1, num_clients + 1):
            client = NLUClientFLTrainer(
                client_id, global_model, client_data_dir, output_dir=output_dir, fl_dir=fl_dir
            )
            client.preprare_local_dataset()
            client_last_layer_fisher_info, client_peft_layer_param = client.select_peft_target_layers(
                tokenizer, percent, compute_metrics, **train_kwargs
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
        selected_peft_config = get_peft_config(peft_name, target_modules)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=glue_task_num_labels[data_name])
        global_model = get_peft_model(model, selected_peft_config)
        
        # using peft_layer_param init the global_model param data
        for peft_layer_name, param in global_model.named_parameters():
            init_peft_layer_param = peft_layer_param.get(peft_layer_name, None)
            if init_peft_layer_param is not None:
                param.data.copy_(init_peft_layer_param.clone())
        
        del trainable_param_old
    
    num_communication_rounds = 30
    eval_metrics_key = glue_task_metrics[data_name][0]
    eval_records = {
        "epches": list(range(1, num_communication_rounds+1)),
        eval_metrics_key: []
    }
    for epoch in tqdm(range(1, num_communication_rounds+1)):
        print(f"\n\nCommunication round {epoch} starts ... ")
        trainable_param_old = get_trainable_param(global_model)
        for client_id in range(1, num_clients + 1):
            client = NLUClientFLTrainer(
                client_id, global_model, client_data_dir, output_dir=output_dir, fl_dir=fl_dir
            )
            print("\nPreparing the local dataset and trainer for Client_{}".format(client_id))
            client.preprare_local_dataset()
            client.build_local_trainer(tokenizer, compute_metrics, add_noise, **train_kwargs)
            
            print("Initiating the local training of Client_{}".format(client_id))
            client.initiate_local_training()
            
            print("Local training starts ... ")
            client.train()
            
            global_model = client.terminate_local_training(trainable_param_old, epoch)
            del client
        
        print("Collecting the weights of clients and performing aggregation")
        global_model = FedAvg(global_model, fl_dir, epoch, num_clients)
        # global_model = circle_aggregation(global_model, fl_dir, epoch, num_clients)
        
        # save the new trainable param
        trainable_param_new = get_trainable_param(global_model)
        torch.save(trainable_param_new, os.path.join(fl_dir, str(epoch), "global_trainable_param.bin"))
        
        # global eval
        eval_result = global_evaluation(global_model, eval_dataset, tokenizer, compute_metrics, output_dir)
        eval_records[eval_metrics_key].append(eval_result[f"eval_{eval_metrics_key}"])
        with open(os.path.join(output_dir, "eval_metrics_records.json"), 'w') as f:
            json.dump(eval_records, f)


def global_evaluation(model, eval_dataset, tokenizer, compute_metrics, output_dir):
    train_args = TrainingArguments(output_dir=output_dir)
    trainer = Trainer(
        model=model, eval_dataset=eval_dataset, args=train_args,
        tokenizer=tokenizer, compute_metrics=compute_metrics
    )
    eval_result = trainer.evaluate()
    return eval_result


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


def circle_aggregation(model, fl_dir, epoch, num_clients=5):
    
    # server init task
    trainable_param = get_trainable_param(model)
    global_pama_keys = trainable_param.keys()
    client_param_keys = {
        client_id: [] for client_id in range(1, num_clients+1)
    }
    # split the global_param_keys to the client_param_keys
    for k, key in enumerate(global_pama_keys):
        client_param_keys[k % num_clients + 1].append(key)
    
    old_client_param = {}
    new_client_param = {}
    cache_client_param = {}
    for k, client_id in enumerate(range(1, num_clients+1)):
        single_output_dir = os.path.join(
            fl_dir, str(epoch), f"local_output_{client_id}", "pytorch_model.bin"
        )
        old_client_param[client_id] = torch.load(single_output_dir)
    for client_id, param_keys in client_param_keys.items():
        cache_client_param[client_id] = {}
        for param_key in param_keys:
            # new_client_param[client_id][param_key] = old_client_param[client_id][param_key].clone()
            # zero the cache_client_param
            cache_client_param[client_id][param_key] = torch.zeros_like(old_client_param[client_id][param_key])
    
    # TODO circle aggregation
    flag = True
    while flag:
        # compute the current client_param using the cache_client_param
        for client_id, cache_params in cache_client_param.items():
            new_client_param[client_id] = {}
            for chache_key, cache_param in cache_params.items():
                new_client_param[client_id][chache_key] = cache_param.clone() + (old_client_param[client_id][chache_key].clone() * (1.0/num_clients))
        
        # clear cache client param using del
        for client_id in range(1, num_clients+1):
            del cache_client_param[client_id]
        
        # send new_client_param to the next client cache_client_param
        for client_id in range(1, num_clients+1):
            next_client_id = client_id + 1 if client_id < num_clients else 1
            cache_client_param[next_client_id] = {}
            for param_key, param in new_client_param[client_id].items():
                cache_client_param[next_client_id][param_key] = param.clone()
        
        # clear new_client_param using del
        for client_id in range(1, num_clients+1):
            del new_client_param[client_id]
        
        # check the flag
        flag = False
        for client_id, cache_params in cache_client_param.items():
            cache_params_keys = cache_params.keys()
            # compare the cache_params with the client_param_keys[client_id], if all elements are the same, then flag = False
            if set(cache_params_keys) != set(client_param_keys[client_id]):
                flag = True
                break
    
    # each client's cache_params contains the partial global_param
    for client_id in range(1, num_clients+1):
        cache_params = cache_client_param[client_id]
        for param_key, param in cache_params.items():
            # send to global server
            trainable_param[param_key] = param.clone()
            # send to all client's new_client_param
            for send_client_id in range(1, num_clients+1):
                new_client_param.setdefault(send_client_id, {})
                new_client_param[send_client_id][param_key] = param.clone()
    
    # for _, cache_params in cache_client_param.items():
    #     for client_id in range(1, num_clients+1):
    #         for param_key, param in cache_params.items():
    #             new_client_param[client_id][param_key] = param.clone()
    
    # client save the new_client_param
    for client_id in range(1, num_clients+1):
        save_path =  os.path.join(fl_dir, str(epoch), f"local_output_{client_id}", "global_pytorch_model.bin")
        torch.save(new_client_param[client_id], save_path)
    
    set_trainable_param(model, trainable_param)
    return model


def modify_client_train_args(train_kwargs):
       train_kwargs['num_train_epochs'] = 1


def set_trainable_param(model, param_dict):
    for name, param in model.named_parameters():
        param_init = param_dict.get(name, None)
        if param_init is not None:
            param.data.copy_(param_init.clone())


def get_trainable_param(model):
    param_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_dict[name] = param.clone().detach()
    return param_dict


def get_peft_config(peft_name: str, target_modules=None):
    
    peft_config = None
    if peft_name == 'lora':
        peft_config = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05,
            task_type=TaskType.SEQ_CLS,
            target_modules=target_modules
        )
    
    assert peft_config is not None, "peft_config is None"
    return peft_config


def get_raw_dataset(data_name, tokenizer, max_length):
    sentence1_key, sentence2_key = glue_task_input_keys[data_name]
    
    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )

        result = tokenizer(*args, max_length=max_length, truncation=True)
        return result

    raw_dataset = load_from_disk(os.path.join(DATA_DIR, "glue", data_name))
    dataset = raw_dataset.map(preprocess_function, batched=True)
    
    train_dataset = dataset["train"].shuffle(42)
    eval_dataset = dataset["validation_matched" if data_name == 'mnli' else 'validation'].shuffle(42)
    return train_dataset, eval_dataset


def prepare_client_dataset(data_name, model_name, tokenizer, max_length, num_clients=5):
    
    client_data_dir = os.path.join(NLU_DIR, "local_data", model_name_full_to_short[model_name], data_name, f"{num_clients}_clients")
    # if os.path.exists(client_data_dir):
    #     return client_data_dir
    # else:
    #     os.makedirs(client_data_dir, exist_ok=True)
    os.makedirs(client_data_dir, exist_ok=True)
    
    # Preprocess dataset
    train_dataset, eval_dataset = get_raw_dataset(data_name, tokenizer, max_length)
    
    train_size = len(train_dataset) // num_clients
    eval_size = len(eval_dataset) // num_clients
    
    for i in range(1, num_clients + 1):
        client_id = i
        
        client_train_dataset = train_dataset[(i-1)*train_size:i*train_size]
        client_train_dataset = Dataset.from_dict(client_train_dataset)
        client_train_dataset.save_to_disk(
            os.path.join(client_data_dir, f"train_dataset_{client_id}")
        )
        
        client_eval_dataset = eval_dataset[(i-1)*eval_size:i*eval_size]
        client_eval_dataset = Dataset.from_dict(client_eval_dataset)
        client_eval_dataset.save_to_disk(
            os.path.join(client_data_dir, f"eval_dataset_{client_id}")
        )

    return client_data_dir, eval_dataset


def get_best_eval_metric(model_name, peft_name, data_name, add_noise, percent):
    # args output-args
    output_dir = get_output_dir(model_name, peft_name, data_name, add_noise, percent)
    if not os.path.exists(os.path.join(output_dir, "eval_metrics_records.json")):
        return None
    with open(os.path.join(output_dir, "eval_metrics_records.json"), 'r') as f:
        eval_result = json.load(f)
    eval_metrics_key = glue_task_metrics[data_name][0]
    assert eval_metrics_key in eval_result, "eval_metrics_key not in eval_result"
    
    # get the best eval result from the list
    best_eval_result = max(eval_result[eval_metrics_key])
    return best_eval_result


def plot_the_eval_metric(model_name, peft_name, data_name, add_noise, percent):
    # args output-args
    output_dir = get_output_dir(model_name, peft_name, data_name, add_noise, percent)
    with open(os.path.join(output_dir, "eval_metrics_records.json"), 'r') as f:
        eval_result = json.load(f)
    eval_metrics_key = glue_task_metrics[data_name][0]
    assert eval_metrics_key in eval_result, "eval_metrics_key not in eval_result"
    
    # plot the eval result, eval_result like {"epches": [1, 2, 3], "eval_mcc": [0.1, 0.2, 0.3]}
    plt.plot(eval_result["epches"], eval_result[eval_metrics_key])
    plt.savefig(os.path.join(output_dir, "eval_metrics_records.png"))
    plt.close()


def get_output_dir(model_name, peft_name, data_name, add_noise, percent):
    # args output-args
    output_dir_prefix = peft_name if peft_name else "base"
    output_dir_suffix = "_dpnoise" if add_noise else ""
    if percent:
        output_dir = os.path.join(
            NLU_DIR, "fl_output", model_name, output_dir_prefix+output_dir_suffix, data_name, "{}_layer".format(percent)
        )
    else:
        output_dir = os.path.join(
            NLU_DIR, "fl_output", model_name, output_dir_prefix+output_dir_suffix, data_name, "full_layer"
        )
    return output_dir


def gather_eval_metrics():
    model_names = nlu_model_names
    peft_names = ['lora']
    exclude_datasets = ['mnli', 'qqp']
    noise_types = [False, True]
    percent_types = [None, 0.7, 0.6, 0.5]
    eval_metrics_dict = {}
    for model_name in model_names:
        key_prefix = model_name_full_to_short[model_name]
        for peft_name in peft_names:
            for add_noise in noise_types:
                for percent in percent_types:
                    if percent is None:
                        key_suffix = f"({peft_name})" + ("_dpnoise" if add_noise else "") + "_fullLayer"
                    else:
                        key_suffix = f"({peft_name})" + ("_dpnoise" if add_noise else "") + f"_{percent}Layer"
                    key = f'{key_prefix}{key_suffix}'
                    eval_metrics_dict[key] = {}
                    for data_name in glue_task_input_keys.keys():
                        if data_name in exclude_datasets:
                            continue
                        best_eval_metric = get_best_eval_metric(model_name, peft_name, data_name, add_noise, percent)
                        if best_eval_metric is not None:
                            plot_the_eval_metric(model_name, peft_name, data_name, add_noise, percent)
                            metric_name = glue_task_metrics[data_name][0]
                            eval_metrics_dict[key].update({
                                f"{data_name}_{metrics_name_full_to_short[metric_name]}": best_eval_metric
                            })
    
    eval_metrics_df = pd.DataFrame(eval_metrics_dict).sort_index(axis=1).T.sort_index(axis=1)
    print(eval_metrics_df)
    eval_metrics_df.to_csv(os.path.join(NLU_DIR, "fl_eval_metrics.csv"))
    

def run():
    model_names = ['roberta-base']
    # model_names = ['bert-base-uncased']
    exclude_datasets = ['mnli', 'qqp']
    noise_types = [False, True]
    # percent_types = [None]
    percent_types = [0.5, 0.6, 0.7]
    for model_name in model_names:
        for data_name in glue_task_input_keys.keys():
            if data_name in exclude_datasets:
                continue
            for add_noise in noise_types:
                for percent in percent_types:
                    print(f"model_name: {model_name}, data_name: {data_name}, add_noise: {add_noise}, percent: {percent}")
                    main(model_name, "lora", data_name, add_noise=add_noise, percent=percent)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="cola")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--peft_name", type=str, default="lora")
    parser.add_argument("--add_noise", type=bool, default=False)
    parser.add_argument("--percent", type=float, default=None)
    args = parser.parse_args()
    print(args)
    
    main(**vars(args))
    plot_the_eval_metric(**vars(args))
    
    # run()
    
    # gather_eval_metrics()
    
    
    