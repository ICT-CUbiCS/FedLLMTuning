import os
import json
import numpy as np
import pandas as pd
import argparse
from datasets import load_from_disk, disable_progress_bar
# disable progress bar
disable_progress_bar()
import evaluate
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    EvalPrediction
)
from peft import (
    LoraConfig, IA3Config,
    get_peft_model, TaskType
)
import adapters

import sys
sys.path.append("..")
from client import ClientWithTrainer, ClientWithAdapterTrainer
from config import (
    PROJECT_DIR, MODEL_DIR, DATA_DIR, NLU_DIR, PEFT_NAMES,
    glue_task_input_keys, glue_task_num_labels, 
    nlu_model_names, evaluates_metric_dir,
    model_name_full_to_short, adapter_peft_names,
    glue_task_metrics, metrics_name_full_to_short
)


def get_output_dir(model_name: str, peft_name: str, data_name: str, add_noise: bool, percent=None):
    output_dir_prefix = peft_name if peft_name else "base"
    output_dir_suffix = "_dpnoise" if add_noise else ""
    if percent:
        output_dir = os.path.join(
            NLU_DIR, "output", model_name, output_dir_prefix+output_dir_suffix, data_name, "{}_layer".format(percent)
        )
    else:
        output_dir = os.path.join(
            NLU_DIR, "output", model_name, output_dir_prefix+output_dir_suffix, data_name, "full_layer"
        )
    return output_dir


def load_model_and_tokenizer(model_name: str, data_name: str, peft_name: str):
    model_path = os.path.join(MODEL_DIR, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=glue_task_num_labels[data_name])
    if peft_name in adapter_peft_names:
        model = get_adapter_model(model, peft_name)
    else:
        peft_config = get_peft_config(peft_name)
        model = get_peft_model(model, peft_config)
    return tokenizer, model


def get_adapter_model(model, adapter_name, leave_out=[]):
    
    adapters.init(model)
    
    if adapter_name == 'bottleneck':
        adapter_config = adapters.BnConfig(
            mh_adapter=True, output_adapter=True,
            reduction_factor=16, non_linearity="relu",
            leave_out=leave_out
        )
    
    model.add_adapter(adapter_name, config=adapter_config)
    model.train_adapter(adapter_name)
    model.set_active_adapters(adapter_name)
    return model


def preprocess_dataset(data_name, tokenizer, max_length):
    sentence1_key, sentence2_key = glue_task_input_keys[data_name]
    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, max_length=max_length, truncation=True)
        return result
    raw_dataset = load_from_disk(os.path.join(DATA_DIR, "glue", data_name))
    dataset = raw_dataset.map(preprocess_function, batched=True)
    return dataset


def prepare_client_args(data_name, client_id, model, output_dir, tokenizer, compute_metrics, add_noise):
    client_kwargs = {}
    # args train-args
    with open(os.path.join(PROJECT_DIR, "task_train_args.json"), "r") as f:
        train_args_dict = json.load(f)
    train_kwargs  = train_args_dict[data_name]
    # disable tqdm
    train_kwargs["disable_tqdm"] = True
    client_kwargs['train_args_dict'] = train_kwargs
    
    client_kwargs.update({
        "client_id": client_id,
        "model": model,
        "output_dir": output_dir,
        
        "data_name": data_name,
        
        "tokenizer": tokenizer,
        "compute_metrics": compute_metrics,
        
        "add_noise": add_noise,
        "dp_epsilon": 0.1   # privacy budget
    })
    
    return client_kwargs


def select_layers(train_dataset, eval_dataset, peft_name, model_name, **client_kwargs):
    
    # Setting up client for layer selection
    client_for_layer_selection = setup_client(peft_name, **client_kwargs)
    client_for_layer_selection.load_local_data(train_dataset, eval_dataset)
    model_path = os.path.join(MODEL_DIR, model_name)
    if peft_name in adapter_peft_names:
        model = get_target_adapter_model(model_path, client_for_layer_selection, peft_name, **client_kwargs)
    else:
        model = get_target_peft_model(model_path, client_for_layer_selection, peft_name, **client_kwargs)
    return model


def get_target_peft_model(model_path, client, peft_name, **client_kwargs):
    client_kwargs.update({"peft_name": peft_name})
    selected_layer_flags, target_layer_param = client.select_target_layers(**client_kwargs)
    selected_peft_config = get_peft_config(peft_name, selected_layer_flags)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=glue_task_num_labels[client_kwargs["data_name"]])
    target_model = get_peft_model(model, selected_peft_config)
    # using target_peft_layer_param init the selected_peft_model param data
    for peft_layer_name, param in target_model.named_parameters():
        init_peft_layer_param = target_layer_param.get(peft_layer_name, None)
        if init_peft_layer_param is not None:
            param.data.copy_(init_peft_layer_param.clone())
    return target_model


def get_target_adapter_model(model_path, client, peft_name, **client_kwargs):
    client_kwargs.update({"peft_name": peft_name})
    unselected_layer_index, target_adapter_layer_param = client.select_target_layers(**client_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=glue_task_num_labels[client_kwargs["data_name"]])
    target_model = get_adapter_model(model, peft_name, leave_out=list(unselected_layer_index))
    for adapter_layer_name, param in model.named_parameters():
        init_adapter_layer_param = target_adapter_layer_param.get(adapter_layer_name, None)
        if init_adapter_layer_param is not None:
            param.data.copy_(init_adapter_layer_param.clone())
    return target_model


def setup_client(peft_name, **client_kwargs):
    if peft_name in adapter_peft_names:
        client = ClientWithAdapterTrainer(**client_kwargs)
    else:
        client = ClientWithTrainer(**client_kwargs)
    return client


def set_up_logging_dir(model_name, peft_name, add_noise, percent, data_name):
    log_dir_name = "{}_{}_{}_{}".format(
        model_name_full_to_short[model_name],
        (f"{peft_name}" if peft_name else "base") + ("_dpnoise" if add_noise else ""),
        "fullLayer" if percent is None else "{}Layer".format(percent),
        data_name
    )
    dir_path = os.path.join(NLU_DIR, "logs", log_dir_name)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path
    

def main(model_name: str, peft_name: str, data_name: str, add_noise: bool = False, percent=None):

    # args output-args
    output_dir = get_output_dir(model_name, peft_name, data_name, add_noise, percent)
    if os.path.exists(os.path.join(output_dir, "client_0", "final_model")):
        print(f"output_dir: {output_dir} already exists")
        return
    
    # Load pretrained model and tokenizer
    tokenizer, model = load_model_and_tokenizer(model_name, data_name, peft_name)
    
    # Preprocess dataset
    max_length = 512
    dataset = preprocess_dataset(data_name, tokenizer, max_length)
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation_matched" if data_name == 'mnli' else 'validation']
    # test_dataset = dataset["test_matched" if data_name == 'mnli' else 'test']
    
    # Get metric function
    metric = evaluate.load(evaluates_metric_dir["glue"], data_name)
    def compute_metrics(pre: EvalPrediction):
        preds = pre.predictions[0] if isinstance(pre.predictions, tuple) else pre.predictions
        # preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        preds = np.squeeze(preds) if glue_task_num_labels[data_name] == 1 else np.argmax(preds, axis=1)
        
        return metric.compute(predictions=preds, references=pre.label_ids)
    
    client_id = 0
    client_kwargs = prepare_client_args(data_name, client_id, model, output_dir, tokenizer, compute_metrics, add_noise)
    if percent is not None:
        client_kwargs["percent"] = percent
        model = select_layers(train_dataset, eval_dataset, peft_name, model_name, **client_kwargs)
        client_kwargs = prepare_client_args(data_name, client_id, model, output_dir, tokenizer, compute_metrics, add_noise)
    
    # Setting up client logging dir
    log_dir = set_up_logging_dir(model_name, peft_name, add_noise, percent, data_name)
    client_kwargs["logging_dir"] = log_dir
    
    client = setup_client(peft_name, **client_kwargs)
    client.load_local_data(train_dataset, eval_dataset)
    client.load_local_trainer(**client_kwargs)
    client.train()
    client.evaluate()


def get_peft_config(peft_name: str, target_modules=None):
    """
    Get peft config by peft_name
    """
    peft_model = None
    
    if peft_name == 'lora':
        peft_model = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05,
            task_type=TaskType.SEQ_CLS, target_modules=target_modules
        )
    
    if peft_name == 'IA3':
        # forwoard module for bert
        feedforward_modules = None
        feedforward_modules_flag = "output.dense"
        if target_modules is not None:
            feedforward_modules = []
            for target_module in target_modules:
                if feedforward_modules_flag in target_module:
                    feedforward_modules.append(target_module)
        peft_model = IA3Config(
            task_type=TaskType.SEQ_CLS,
            target_modules=target_modules,
            feedforward_modules=feedforward_modules,
        )
    
    assert peft_model is not None, f"peft_name: {peft_name} is not supported"
    return peft_model


def gather_eval_results():
    # Define the parameters
    model_names = nlu_model_names
    peft_names = PEFT_NAMES
    data_names = list(glue_task_input_keys.keys())
    noise_types = [False, True]
    percents = [None, 0.5, 0.6, 0.7]

    # Initialize the result dictionary
    eval_result_dict = {}

    # Loop over all combinations of parameters
    for model_name in model_names:
        key_prefix = model_name_full_to_short[model_name]
        for peft_name in peft_names:
            for add_noise in noise_types:
                key_middle = peft_name + ("_dpnoise" if add_noise else "")
                for percent in percents:
                    key_subfix = "fullLayer" if percent is None else "{}Layer".format(percent)
                    key = "{}_{}_{}".format(key_prefix, key_middle, key_subfix)

                    # Initialize the dictionary for this combination of parameters if it doesn't exist
                    if key not in eval_result_dict:
                        eval_result_dict[key] = {}

                    for data_name in data_names:
                        # Construct the path to the evaluation metrics file
                        output_dir = get_output_dir(model_name, peft_name, data_name, add_noise, percent)
                        final_model_dir = os.path.join(output_dir, "client_0", "final_model")
                        eval_metrics_file = os.path.join(final_model_dir, "eval_metrics.json")

                        # If the file exists, load the evaluation metrics and add them to the result dictionary
                        if os.path.exists(eval_metrics_file):
                            with open(eval_metrics_file, "r") as f:
                                eval_name = glue_task_metrics[data_name][0]
                                file_dict = json.load(f)
                                target_result = file_dict[f"eval_{eval_name}"]
                                eval_result_dict[key].update({
                                    f"{data_name}_{metrics_name_full_to_short[eval_name]}": target_result
                                })

    # Convert the result dictionary to a DataFrame and sort it
    metrics_df = pd.DataFrame(eval_result_dict).sort_index(axis=1).T.sort_index(axis=1)
    metrics_df.to_csv(os.path.join(NLU_DIR, "eval_results.csv"))
    print(metrics_df)

    return metrics_df


def main_test():
    
    # model_name = "bert-base-uncased"
    model_name = "roberta-base"
    
    peft_name = "lora"
    # peft_name = "IA3"
    # peft_name = "bottleneck" # adapter
    
    data_name = "cola"
    
    # Trainer or AdapterTrainer
    main(model_name, peft_name, data_name, add_noise=False, percent=None)
    
    # noise with Trainer or AdapterTrainer
    main(model_name, peft_name, data_name, add_noise=True, percent=None)
    
    # layer selection with Trainer or AdapterTrainer
    main(model_name, peft_name, data_name, add_noise=False, percent=0.7)
    
    # layer_selection and noise with Trainer or AdapterTrainer
    main(model_name, peft_name, data_name, add_noise=True, percent=0.7)


if __name__ == "__main__":
    
    # main_test()
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name", type=str, required=True)
    # parser.add_argument("--peft_name", type=str, required=True)
    # parser.add_argument("--data_name", type=str, required=True)
    # parser.add_argument("--add_noise", action="store_true", default=False)
    # parser.add_argument("--percent", type=float, default=None)
    
    # args = parser.parse_args()
    # print("\nargs:")
    # print(args)
    # main(args.model_name, args.peft_name, args.data_name, args.add_noise, args.percent)
    
    gather_eval_results()
