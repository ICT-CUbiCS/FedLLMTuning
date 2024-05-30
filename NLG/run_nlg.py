import os
import argparse
import json
from typing import Dict
from tqdm import tqdm
import pandas as pd
import torch
import adapters
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    StoppingCriteriaList
)
# from transformers.integrations import TensorBoardCallback
import evaluate
from datasets import load_from_disk, disable_progress_bar
disable_progress_bar()
from peft import (
    LoraConfig, PrefixTuningConfig, IA3Config,
    AutoPeftModelForCausalLM,
    get_peft_model, TaskType,
    AdaLoraConfig
)

from train_e2e import get_dataset
from generate_stratege import StopAtSpecificTokenCriteria

import sys
sys.path.append("..")
from client import NLGClientWithSFTTrainer, NLGClientWithTrainer
from client import ClientWithTrainer, ClientWithAdapterTrainer
from config import (
    PROJECT_DIR, MODEL_DIR, DATA_DIR, NLG_DIR, PEFT_NAMES, DP_NOISE,
    nlg_dataset_to_keys, nlg_model_names, adapter_peft_names, model_name_full_to_short
)
from e2e_metrics.measure_scores import load_data, evaluate

def get_output_dir(model_name: str, peft_name: str, data_name: str, add_noise: bool, percent=None):
    output_dir_prefix = peft_name if peft_name else "base"
    output_dir_suffix = "_dpnoise" if add_noise else ""
    if percent:
        output_dir = os.path.join(
            NLG_DIR, "output", model_name, output_dir_prefix+output_dir_suffix, data_name, "{}_layer".format(percent)
        )
    else:
        output_dir = os.path.join(
            NLG_DIR, "output", model_name, output_dir_prefix+output_dir_suffix, data_name, "full_layer"
        )
    return output_dir


def load_model_and_tokenizer(model_name: str, peft_name: str):
    model_path = os.path.join(MODEL_DIR, model_name)
    model, tokenizer = get_model(model_path)
    assert model is not None, f"model: {model} is None"
    if peft_name in adapter_peft_names:
        model = get_adapter_model(model, peft_name)
    else:
        peft_config = get_peft_config(peft_name, model_name)
        assert peft_config is not None, "peft_config is None"
        model = get_peft_model(model, peft_config)
    return tokenizer, model


def get_adapter_model(model, adapter_name, leave_out=[]):

    adapters.init(model)

    if adapter_name == 'bottleneck':
        config = adapters.BnConfig(
            mh_adapter=True, output_adapter=True,
            reduction_factor=16, non_linearity="relu",
            leave_out=leave_out
        )

    model.add_adapter(adapter_name, config=config)
    model.train_adapter(adapter_name)
    model.set_active_adapters(adapter_name)
    return model


def preprocess_dataset(data_name, tokenizer, max_length):
    train_dataset, eval_dataset = get_dataset(data_name)
    train_dataset = train_dataset.shuffle(seed=42)
    sentence1_key, sentence2_key = nlg_dataset_to_keys[data_name]
    
    def preprocess_function(examples):
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


def prepare_client_args(data_name, model_name, peft_name, client_id, model, output_dir, tokenizer, add_noise):
    client_kwargs = {}
    with open(os.path.join(PROJECT_DIR, "task_train_args.json"), "r") as f:
        train_args_dict = json.load(f)
    train_kwargs  = train_args_dict[data_name][model_name]
    if peft_name in ['IA3', 'bottleneck']:
        train_kwargs.update({
            "num_train_epochs": 30,
            
            # "load_best_model_at_end": True,
            # "metric_for_best_model": "eval_loss"
        })
    # disable tqdm
    train_kwargs["disable_tqdm"] = True
    
    client_kwargs['train_args_dict'] = train_kwargs
    client_kwargs.update({
        "client_id": client_id,
        "model": model,
        "output_dir": output_dir,
        
        "tokenizer": tokenizer,
        
        "add_noise": add_noise,
        "dp_epsilon": 0.15
    })
    
    return client_kwargs


def set_up_logging_dir(model_name, peft_name, add_noise, percent, data_name):
    log_dir_name = "{}_{}_{}_{}".format(
        model_name_full_to_short[model_name],
        (f"{peft_name}" if peft_name else "base") + ("_dpnoise" if add_noise else ""),
        "fullLayer" if percent is None else "{}Layer".format(percent),
        data_name
    )
    dir_path = os.path.join(NLG_DIR, "logs", log_dir_name)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

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
    selected_peft_config = get_peft_config(peft_name, model_path, selected_layer_flags)
    model, _ = get_model(model_path)
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
    model, _ = get_model(model_path)
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


def prepare_eval_peft_model(final_model_path):
    
    if "gpt" in final_model_path:
        model = AutoPeftModelForCausalLM.from_pretrained(final_model_path)
        model.to("cuda")
    if "llama" in final_model_path or "LLaMA" in final_model_path:
        model = AutoPeftModelForCausalLM.from_pretrained(
            final_model_path,
            torch_dtype=torch.bfloat16,
            load_in_4bit=True
        )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(final_model_path, padding_side="left")
    return model, tokenizer


def prepare_eval_adapter_model(final_model_path):
    with open(os.path.join(final_model_path, "adapter_config.json"), "r") as f:
        adapter_config = json.load(f)
        pretrained_model_name = adapter_config["model_name"]
    model, tokenizer = get_model(pretrained_model_name)
    tokenizer.padding_side = "left"
    adapters.init(model)
    model.load_adapter(final_model_path)
    model.set_active_adapters(adapter_config['name'])
    model.to("cuda")
    model.eval()
    return model, tokenizer


def prepare_eval_dataset(data_name):
    _, eval_dataset = get_dataset(data_name)
    
    def query_string(meaning_representation):
        # return f"""meaning_representation: {meaning_representation}\nhuman_reference: """
        return f"""{meaning_representation}\n"""

    def query_string_process(sample):
        result = {"prompt": []}
        for meaning_representation in sample["meaning_representation"]:
            result["prompt"].append(query_string(meaning_representation))
        return result
    
    eval_dataset = eval_dataset.map(query_string_process, batched=True)
    prompts = eval_dataset["prompt"]
    meaning_representations = eval_dataset["meaning_representation"]
    human_references = eval_dataset["human_reference"]
    
    # save the refer_dict
    refer_dict = {}
    for meaning_representation, human_reference in zip(meaning_representations, human_references):
        if meaning_representation not in refer_dict:
            refer_dict[meaning_representation] = {}
            refer_dict[meaning_representation]['human_reference'] = []
        refer_dict[meaning_representation]['human_reference'].append(human_reference)
    
    return prompts, meaning_representations, refer_dict


def eval_generate(final_model_path, model, tokenizer, prompts, meaning_representations, refer_dict, batch_size=1):
    
    prompt_file_path = os.path.join(final_model_path, "prompt.txt")
    reference_file_path = os.path.join(final_model_path, "reference.txt")
    generate_file_path = os.path.join(final_model_path, "generate.txt")
    
    prompt_file = open(prompt_file_path, "w")
    references_file = open(reference_file_path, "w")
    generates_file = open(generate_file_path, "w")
    
    stop_token_ids = [
        tokenizer.pad_token_id, tokenizer.eos_token_id,
        tokenizer("name").input_ids[-1], 
        tokenizer(".").input_ids[-1]
    ]
    stop_criteria = StopAtSpecificTokenCriteria(stop_token_ids)
    criteria_list = StoppingCriteriaList([stop_criteria])
    
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        batch_meaning_representations = meaning_representations[i:i+batch_size]
        batch_inputs = tokenizer(batch_prompts, return_tensors="pt", padding='longest', truncation=True, max_length=256).to(model.device)
        batch_outputs = model.generate(
            **batch_inputs, pad_token_id = tokenizer.eos_token_id, 
            max_new_tokens=64, do_sample=True,
            num_beams=10,
            stopping_criteria=criteria_list
        )
        generate_strings = tokenizer.batch_decode(
            batch_outputs.detach().cpu().numpy(), skip_special_tokens=True
        )
        for j, prompt in enumerate(batch_prompts):
            meaning_representation = batch_meaning_representations[j]
            generate_string = generate_strings[j][len(prompt):].strip().replace("\n", "")
            references = refer_dict.get(meaning_representation, {}).get('human_reference', [])
            if len(generate_string) > 10 and len(references) > 0:
                prompt_file.write(prompt.replace("\n", "") + "\n")
                generates_file.write(generate_string + "\n")
                for ref in references:
                    references_file.write(ref + "\n")
                references_file.write("\n")
    
    # close file
    references_file.close()
    generates_file.close()
    prompt_file.close()
    
    return reference_file_path, generate_file_path


def eval_with_generate(final_model_path, data_name, peft_name):
    
    if peft_name in adapter_peft_names:
        model, tokenizer = prepare_eval_adapter_model(os.path.join(final_model_path, peft_name))
    else:
        model, tokenizer = prepare_eval_peft_model(final_model_path)
    
    prompts, meaning_representations, refer_dict = prepare_eval_dataset(data_name)
    reference_file_path, generate_file_path = eval_generate(final_model_path, model, tokenizer, prompts, meaning_representations, refer_dict)
    
    data_src, data_ref, data_sys = load_data(reference_file_path, generate_file_path)
    scores = evaluate(data_src, data_ref, data_sys, python=True)
    print(scores)
    with open(os.path.join(final_model_path, "metrics_scores.json"), "w") as f:
        json.dump(scores, f, indent=2)


def train_with_trainer(model_name: str, peft_name: str, data_name: str, add_noise: bool = False, percent: float = None):
    
    # args output-args
    output_dir = get_output_dir(model_name, peft_name, data_name, add_noise, percent)
    
    # model setup
    tokenizer, model = load_model_and_tokenizer(model_name, peft_name)
    
    # dataset process
    max_length = 512
    train_dataset, eval_dataset = preprocess_dataset(data_name, tokenizer, max_length)    
    
    client_id = 0
    client_kwargs = prepare_client_args(data_name, model_name, peft_name, client_id, model, output_dir, tokenizer, add_noise)
    if percent is not None:
        client_kwargs["percent"] = percent
        # reduce the num of train_dataset
        train_dataset_small = train_dataset.select(range(int(len(train_dataset)*0.1))) 
        model = select_layers(train_dataset_small, eval_dataset, peft_name, model_name, **client_kwargs)
        client_kwargs = prepare_client_args(data_name, model_name, peft_name, client_id, model, output_dir, tokenizer, add_noise)

    log_dir = set_up_logging_dir(model_name, peft_name, add_noise, percent, data_name)
    client_kwargs["logging_dir"] = log_dir
    
    client = setup_client(peft_name, **client_kwargs)
    client.load_local_data(train_dataset, eval_dataset)
    client.load_local_trainer(**client_kwargs)
    client.train()
    
    #TODO eval_generate for peft and adapter !!!
    eval_with_generate(client.final_model_dir, data_name, peft_name)
    # eval_generate(client.final_model_dir, data_name, model_name)


def train_with_sfttrainer(
    model_name: str, peft_name: str, data_name: str, 
    add_noise=False, percent = None
):
    
    # model and dataset dir path
    model_path = os.path.join(MODEL_DIR, model_name)
    data_path = os.path.join(DATA_DIR, data_name)
    
    # args output-args
    output_dir_prefix = peft_name if peft_name else "base"
    output_dir_suffix = "_dp_noise" if add_noise else ""
    if percent:
        output_dir = os.path.join(
            NLG_DIR, "output", model_name, output_dir_prefix+output_dir_suffix, data_name, "{}_layer".format(percent)
        )
    else:
        output_dir = os.path.join(
            NLG_DIR, "output", model_name, output_dir_prefix+output_dir_suffix, data_name, "full_layer"
        )
    
    # args train-args
    with open(os.path.join(PROJECT_DIR, "task_train_args.json"), "r") as f:
        train_args_dict = json.load(f)
    train_kwargs  = train_args_dict[data_name][model_name]
    
    # model setup
    model, tokenizer = get_model(model_path)
    assert model is not None, f"model: {model} is None"
    if peft_name:
        peft_config = get_peft_config(peft_name, model_name)
        assert peft_config is not None, "peft_config is None"
        peft_model = get_peft_model(model, peft_config)
    
    # train and validate data setup
    raw_dataset = load_from_disk(data_path)
    print("\nraw_dataset:\n", raw_dataset)
    train_dataset = raw_dataset["train"].shuffle(seed=42)
    eval_dataset = raw_dataset["validation"].shuffle(seed=42)
    
    # training client setup
    training_client = NLGClientWithSFTTrainer(
        client_id = 0, model = peft_model, output_dir = output_dir
    )
    training_client.load_local_data(train_dataset, eval_dataset)
    
    # format func
    format_func, _ = get_format_func(data_name)
    
    # whether or not use peft layer selection
    if peft_name and percent:
        target_module_names, target_peft_layer_param =training_client.selecte_peft_target_layers(
            tokenizer, percent, format_func=format_func, **train_kwargs
        )
        for target_module_name in target_module_names:
            print(target_module_name)
        selected_peft_config = get_peft_config(peft_name, model_name, target_module_names)
        model, tokenizer = get_model(model_path)
        selected_peft_layer_model = get_peft_model(model, selected_peft_config)
        for peft_layer_name, param in selected_peft_layer_model.named_parameters():
            init_peft_layer_param = target_peft_layer_param.get(peft_layer_name, None)
            if init_peft_layer_param is not None:
                param.data.copy_(init_peft_layer_param.clone())
        del training_client
        training_client = NLGClientWithSFTTrainer(
            client_id = 0, model = selected_peft_layer_model, output_dir = output_dir
        )
        training_client.load_local_data(train_dataset, eval_dataset)
    
    training_client.load_local_trainer(
        tokenizer=tokenizer, format_func=format_func, add_noise=add_noise, **train_kwargs
    )
    training_client.train()


def get_format_func(data_name):
    
    if data_name == "e2e_nlg":
        def format_string(sample):
            # return f"""meaning_representation: {sample['meaning_representation']}\nhuman_reference: {sample['human_reference']}"""
            return f"""{sample['meaning_representation']}\n{sample['human_reference']}"""

        def query_string(meaning_representation):
            # return f"""meaning_representation: {meaning_representation}\nhuman_reference: """
            return f"""{meaning_representation}\n"""
        return format_string, query_string


def get_peft_config(peft_name: str, model_name: str, target_modules=None):
    """Get peft config."""
    # assert peft_name in PEFT_NAMES, f"peft_name: {peft_name} is not in {PEFT_NAMES}"
    peft_config = None
    if peft_name == "lora":
        if "gpt" in model_name:
            peft_config = LoraConfig(
                r=4,
                lora_alpha=32,
                lora_dropout=0.1,
                fan_in_fan_out=True,
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules
            )
        if "llama" in model_name or "LLaMA" in model_name:
            peft_config = LoraConfig(
                r=24,
                lora_alpha=48,
                lora_dropout=0.1,
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules
            )
    if peft_name == 'ada_lora':
        peft_config = AdaLoraConfig(
            lora_alpha=32, lora_dropout=0.1,
            target_r=16, init_r=16,
            task_type=TaskType.CAUSAL_LM
        )
    if peft_name == "prefixtuning":
        peft_config = PrefixTuningConfig(
            num_virtual_tokens=20,
            task_type=TaskType.CAUSAL_LM
        )
    if peft_name == "IA3":
        feedforward_modules = None
        if "gpt2" in model_name:
            feedforward_modules_flag = 'mlp.c_proj'
        if "llama" in model_name:
            feedforward_modules_flag = 'down_proj'
        if target_modules is not None:
            feedforward_modules = []
            for target_module in target_modules:
                if feedforward_modules_flag in target_module:
                    feedforward_modules.append(target_module)
            assert len(feedforward_modules) > 0, \
                "feedforward_modules should not be empty if target_modules is not None"
        peft_config = IA3Config(
            task_type=TaskType.CAUSAL_LM,
            fan_in_fan_out=True if "gpt" in model_name else False,
            target_modules=target_modules,
            feedforward_modules=feedforward_modules,
        )
    return peft_config


def trainning_data_process(data_name, peft_name, model, tokenizer, raw_dataset):
    """Process training data for training."""
    max_length = 512
    if peft_name == "prefixtuning":
        max_length = max_length - model.active_peft_config.num_virtual_tokens
    
    sentence1_key, sentence2_key = nlg_dataset_to_keys[data_name]
    # def preprocess_function(examples):
    #     result = tokenizer(
    #         [" ".join([example1, example2 + tokenizer.eos_token]) 
    #             for example1, example2 in zip(examples[sentence1_key], examples[sentence2_key])]
    #     )
    #     return result
    def preprocess_function(examples):
        result = tokenizer(
            [" ".join([example1, example2 + tokenizer.eos_token]) 
                for example1, example2 in zip(examples[sentence1_key], examples[sentence2_key])]
        )
        return result
    
    tokenizer_data = raw_dataset.map(
        preprocess_function, batched=True, remove_columns=raw_dataset['train'].column_names, num_proc=4
    )
    print("\ntokenizer_data:\n", tokenizer_data)
    
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
    
    group_data = tokenizer_data.map(group_texts, batched=True, num_proc=4)
    train_dataset = group_data['train']
    eval_dataset = group_data['validation']
    print("\ntrain_dataset[0] strs:\n", tokenizer.decode(train_dataset[0]['input_ids']))
    
    return train_dataset, eval_dataset


def train(model_name, data_name, output_dir, quantization=False, peft_name='lora'):
    
    model_path = os.path.join(MODEL_DIR, model_name)
    data_path = os.path.join(DATA_DIR, data_name)
    assert os.path.exists(model_path) , f"model path: {model_path} is not exist"
    assert os.path.exists(data_path), f"data path: {data_path} is not exist"
    
    # model setup
    model, tokenizer = get_model(model_path, quantization=quantization)
    if peft_name:
        peft_config = get_peft_config(peft_name)
        model = get_peft_model(model, peft_config)
    
    # train and validate data setup
    raw_dataset = load_from_disk(data_path)
    raw_dataset = raw_dataset.shuffle(seed=42)
    print("\nraw_dataset:\n", raw_dataset)
    
    train_dataset, eval_dataset = trainning_data_process(
        data_name, peft_name, model, tokenizer, raw_dataset
    )
    
    # data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # import numpy as np
    # from transformers import EvalPrediction
    # metric = evaluate.load('rouge')
    # def compute_metrics(pre: EvalPrediction):
    #     predictions = np.where(pre.predictions != -100, pre.predictions, tokenizer.pad_token_id)
    #     predictions_strs = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    #     references = np.where(pre.label_ids != -100, pre.label_ids, tokenizer.pad_token_id)
    #     references_strs = tokenizer.batch_decode(references, skip_special_tokens=True)
    #     rouge_output = metric.compute(predictions=predictions_strs, references=references_strs)
    #     return rouge_output
    
    # from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
    # args = Seq2SeqTrainingArguments(
    #     output_dir=output_dir,
    #     num_train_epochs=1,
    #     per_device_train_batch_size=16,
    #     per_device_eval_batch_size=16,
    #     evaluation_strategy="steps",
    #     save_strategy="steps",
    #     logging_dir=os.path.join(output_dir, "logs"),
    #     logging_steps=10,
    #     eval_steps=10,
    #     save_steps=10,
    #     save_total_limit=1,
    #     predict_with_generate=True,
    # )
    # trainer = Seq2SeqTrainer(
    #     model=model,
    #     args=args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     # compute_metrics=compute_metrics,
    #     tokenizer=tokenizer,
    #     data_collator=data_collator
    # )
    
    args = get_training_args(model_name, output_dir)
    
    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    try:
        trainer.train()
        trainer.save_model(os.path.join(output_dir, "final"))
    except KeyboardInterrupt:
        trainer.save_model(os.path.join(output_dir, "interrupt"))
        raise KeyboardInterrupt


def get_model(model_path):
    model = None
    if "gpt" in model_path:
        model = AutoModelForCausalLM.from_pretrained(model_path)
    if "llama" in model_path or "LLaMA" in model_path:
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
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_training_args(model_name, output_dir):

    assert model_name in nlg_model_names, f"model_name: {model_name} is not in {nlg_model_names}"
    from transformers import TrainingArguments
    if "gpt" in model_name:
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=5, learning_rate=2e-4,
            per_device_train_batch_size=12, gradient_accumulation_steps=4,
            per_device_eval_batch_size=12, eval_accumulation_steps=4,
            evaluation_strategy="epoch", save_strategy="epoch",
            # evaluation_strategy="steps", eval_steps=10,
            # save_strategy="steps", save_steps=10,
            logging_dir=os.path.join(output_dir, "logs"), logging_steps=10,
            save_total_limit=1,
        )
    if "llama" in model_name or "LLaMA" in model_name:
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=5, optim="paged_adamw_32bit", 
            learning_rate=2e-4, lr_scheduler_type="constant",
            per_device_train_batch_size=8, gradient_accumulation_steps=4,
            per_device_eval_batch_size=8, eval_accumulation_steps=4,
            evaluation_strategy="epoch", save_strategy="epoch",
            logging_dir=os.path.join(output_dir, "logs"), logging_steps=10,
            save_total_limit=1,
            fp16=True,
            warmup_ratio=0.1,
        )
    
    return args


def test_generate(output_dir, data_name, quantization=False, batch_size=32):
    """Generate text and evaluate rouge score."""
    # tokenizer = AutoTokenizer.from_pretrained(output_dir, padding_side="left")
    # tokenizer.pad_token = tokenizer.eos_token
    
    # model = AutoModelForCausalLM.from_pretrained(output_dir)
    model, tokenizer = get_model(output_dir, quantization=quantization)
    tokenizer.padding_side = "left"
    # tokenizer.pad_token = tokenizer.eos_token
    if not quantization:    # warning: `.to` is not supported for `4-bit` or `8-bit` bitsandbytes models
        model.to("cuda")
    model.eval()
    
    sentence1_key, sentence2_key = nlg_dataset_to_keys[data_name]
    test_dataset = load_from_disk(os.path.join(DATA_DIR, data_name))["test"].select(range(100))
    input_texts = test_dataset[sentence1_key]
    target_texts = test_dataset[sentence2_key]
    
    output_texts = []
    for i in tqdm(range(0, len(input_texts), batch_size)):
        batch_input_texts = input_texts[i:i+batch_size]
        batch_target_texts = target_texts[i:i+batch_size]
        batch_input = tokenizer(
            batch_input_texts, return_tensors="pt", padding='longest', truncation=True, max_length=256
        ).to(model.device)
        batch_target = tokenizer(
            batch_target_texts, return_tensors="pt", padding='longest', truncation=True, max_length=256
        ).to(model.device)
        with torch.no_grad():
            batch_output_ids = model.generate(
                **batch_input, pad_token_id = tokenizer.eos_token_id, 
                max_length = batch_target.input_ids.shape[1]+batch_input.input_ids.shape[1]
                # max_length = 512
            )
        batch_output_texts = tokenizer.batch_decode(batch_output_ids, skip_special_tokens=True)
        output_texts.extend(batch_output_texts)
    
    output_texts = [
        output_text[len(input_text):].replace('\n', '') for input_text, output_text in zip(input_texts, output_texts)
    ]
    
    with open("input_tmp.txt", "w") as f:
        f.write("\n".join(input_texts))
    with open("output_tmp.txt", "w") as f:
        f.write("\n".join(output_texts))
    with open("target_tmp.txt", "w") as f:
        f.write("\n".join(target_texts))
    
    # rouge_output = evaluate.load('rouge').compute(predictions=output_texts, references=target_texts)
    # print(rouge_output)


def run(model_name, quantization=False):
    for peft_name in PEFT_NAMES:
        for data_name in nlg_dataset_to_keys.keys():
            for noise in DP_NOISE:
                print(f"\nmodel_name: {model_name}, peft_name: {peft_name}, data_name: {data_name}, noise: {noise}:\n")
                train_with_sfttrainer(model_name, peft_name, data_name, quantization=quantization, noise=noise)
                torch.cuda.empty_cache()



def gather_eval_results():
    # Define the parameters
    model_names = nlg_model_names
    peft_names = PEFT_NAMES
    data_names = ['e2e_nlg']
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

                    for data_name in data_names:
                        # Construct the path to the evaluation metrics file
                        output_dir = get_output_dir(model_name, peft_name, data_name, add_noise, percent)
                        final_model_dir = os.path.join(output_dir, "client_0", "final_model")
                        eval_metrics_file = os.path.join(final_model_dir, "metrics_scores.json")

                        # If the file exists, load the evaluation metrics and add them to the result dictionary
                        if os.path.exists(eval_metrics_file):
                            with open(eval_metrics_file, 'r') as f:
                                eval_result = json.load(f)
                            eval_result_dict[key] = eval_result

    # Convert the result dictionary to a DataFrame and sort it
    metrics_df = pd.DataFrame(eval_result_dict).sort_index(axis=1).T.sort_index(axis=1)
    metrics_df.to_csv(os.path.join(NLG_DIR, "eval_results.csv"))
    print(metrics_df)

    return metrics_df


def main_test():
    
    # model_name = "gpt2-medium"
    model_name = "llama2-7b"
    
    # peft_name = "lora"
    peft_name = "IA3"
    # peft_name = "bottleneck" # adapter
    
    data_name = "e2e_nlg"
    
    # Trainer or AdapterTrainer
    train_with_trainer(model_name, peft_name, data_name, add_noise=False, percent=None)
    
    # noise with Trainer or AdapterTrainer
    train_with_trainer(model_name, peft_name, data_name, add_noise=True, percent=None)
    
    # layer selection with Trainer or AdapterTrainer
    train_with_trainer(model_name, peft_name, data_name, add_noise=False, percent=0.7)

def print_model_size(model_name, peft_name, data_name, add_noise, percent):
    output_dir = get_output_dir(model_name, peft_name, data_name, add_noise, percent)
    final_model_path = os.path.join(output_dir, "client_0", "final_model")

    if "gpt" in final_model_path:
        model = AutoPeftModelForCausalLM.from_pretrained(final_model_path)
        model.to("cuda")
    if "llama" in final_model_path or "LLaMA" in final_model_path:
        model = AutoPeftModelForCausalLM.from_pretrained(
            final_model_path,
            torch_dtype=torch.bfloat16,
            load_in_4bit=True
        )

    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for name, param in model.named_parameters():
            all_param += param.numel()
            #if param.requires_grad:
            if 'lora' in name:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    print(model_name, peft_name, percent)
    print_trainable_parameters(model)
    print()


if __name__ == "__main__":
    
    # main_test()
    
    # get args from command line using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--peft_name", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--add_noise", action="store_true", default=False)
    parser.add_argument("--percent", type=float, default=None)
    
    args = parser.parse_args()
    print("\nargs:")
    print(args)
    
    #train_with_trainer(**vars(args))
    # output_dir = get_output_dir(
    #     args.model_name, args.peft_name, args.data_name, args.add_noise, args.percent
    # )
    # assert os.path.exists(output_dir), f"output_dir: {output_dir} is not exist"
    # eval_with_generate(
    #     final_model_path=os.path.join(output_dir, "client_0", "final_model"),
    #     data_name=args.data_name, peft_name=args.peft_name
    # )
    
    # gather_eval_results()

    print_model_size(**vars(args))
