import json
import os
import re

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

from transformers.trainer_callback import TrainerCallback


class AdapterLayerFisherInfoCallback(TrainerCallback):
    
    def __init__(self, output_dir, adapter_name, percent=0.7, accumulate_grads=True):
        # working directory
        self.output_dir = output_dir
        
        # middle variables
        self.grads = {}  # Dictionary to store the gradients
        self.temp_params = {}  # Temporary dictionary to store the parameters at the beginning of each step
        self.num_steps = 0
        self.accumulate_grads = accumulate_grads
        
        self.epoch_layer_avg_grads = {}
        self.unselected_layers = set()
        self.target_adapter_layer_param = {}
        self.adapter_name = adapter_name
        self.percent = percent
        
    def on_train_begin(self, args, state, control, **kwargs):
        
        model = kwargs.get('model', None)
        assert model is not None, "model is None"
        
        adapter_layer_names = []
        for name, param in model.named_parameters():
            if param.requires_grad and 'classifier' not in name:
                self.grads[name] = torch.zeros_like(param).to("cpu")
                adapter_layer_names.append(name)
        
        for layer_index in self.get_adapter_layer_index(adapter_layer_names):
            self.epoch_layer_avg_grads[layer_index] = []

    def on_train_end(self, args, state, control, **kwargs):
        """plot fisher info"""
        layer_fisher_info_df = pd.DataFrame(self.epoch_layer_avg_grads)
        
        # plot the df line fig
        self.plot_line_fig(layer_fisher_info_df)
        
        # plot the df box fig
        self.plot_box_fig(layer_fisher_info_df)
        
        # plot the heatmap using the cos similar bettwen selected layer one hot code in all epoch
        layer_rankd_df = layer_fisher_info_df.rank(axis=1, ascending=False)
        layer_one_hot_df = layer_rankd_df.apply(
            lambda row: row <= int(len(layer_rankd_df.columns) * self.percent), axis=1
        )
        self.plot_heatmap_fig(layer_one_hot_df)
        
        # update the un-selected layer index
        last_one_hot = layer_one_hot_df.loc[len(layer_one_hot_df) - 1]
        last_unselected_layers = last_one_hot[last_one_hot==0].index.tolist()
        for layer_index in last_unselected_layers:
            self.unselected_layers.add(layer_index)
        
        # save the adapter layer param
        model = kwargs.get("model", None)
        last_selected_layers = last_one_hot[last_one_hot==1].index.tolist()
        for layer_index in last_selected_layers:
            for adapter_layer_name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if f".{layer_index}.attention" in adapter_layer_name or f".{layer_index}.output" in adapter_layer_name:
                # if layer_index in adapter_layer_name:
                    self.target_adapter_layer_param[adapter_layer_name] = param.clone().detach()

    def on_epoch_end(self, args, state, control, **kwargs):
        
        avg_grads = {}
        for name, param in self.grads.items():
            tmp_param = param.clone().detach().to("cpu")
            avg_grads[name] = tmp_param / self.num_steps
        
        # compute the fisher info of each layer
        layer_fisher_info = {}
        for name, avg_grad in avg_grads.items():
            layer_index = list(self.get_adapter_layer_index([name]))[0]
            layer_fisher_info[layer_index] = layer_fisher_info.get(layer_index, 0) + float(torch.mean(avg_grad))
        
        for layer_index, fisher_info in layer_fisher_info.items():
            self.epoch_layer_avg_grads[layer_index].append(fisher_info)
        
        # save the epoch_layer_avg_grads
        with open(os.path.join(self.output_dir, "epoch_layer_avg_grads.json"), "w") as f:
            json.dump(self.epoch_layer_avg_grads, f, indent=4)
        
        # do not accumulate the grads
        if not self.accumulate_grads:
            for name, param in self.grads.items():
                self.grads[name] = torch.zeros_like(param).to("cpu")
            self.num_steps = 0

    def on_step_begin(self, args, state, control, **kwargs):
        """Store the parameters at the beginning of each step"""
        model = kwargs.get('model', None)
        assert model is not None, "model is None"
        for name, param in model.named_parameters():
            if param.requires_grad and 'classifier' not in name:
                self.temp_params[name] = param.clone().detach().to("cpu")

    def on_step_end(self, args, state, control, **kwargs):
        """Calculate the gradient of each trainable parameter and accumulate it"""
        model = kwargs.get('model', None)
        assert model is not None, "model is None"
        for name, param in model.named_parameters():
            if param.requires_grad and 'classifier' not in name:
                tmp_param = param.clone().detach().to("cpu")
                self.grads[name] += (tmp_param - self.temp_params[name]) ** 2
        
        self.num_steps += 1 
    
    def get_adapter_layer_index(self, adapter_layer_names):
        
        pattern = rf"\.(\d+)\."
        index_set = set()
        if self.adapter_name == 'bottleneck':
            for adapter_layer_name in adapter_layer_names:
                match = re.search(pattern, adapter_layer_name)
                if match:
                    index_set.add(int(match.group(1)))
        elif self.adapter_name == 'prefix':
            pass
        
        assert len(index_set) > 0, "index_set is empty"
        return index_set
    
    def plot_line_fig(self, layer_fisher_info_df):
        ax = layer_fisher_info_df.plot(figsize=(16, 8))
        plt.savefig(os.path.join(self.output_dir, "layer_fisher_info_df.png"))
        plt.close()

    def plot_box_fig(self, layer_fisher_info_df):
        ax = layer_fisher_info_df.boxplot(figsize=(16, 8))
        plt.xticks(rotation='vertical')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "layer_fisher_info_df_box.png"))
        plt.close()
    
    def plot_heatmap_fig(self, layer_one_hot_df):
        
        cosine_similarity_matrix = get_cosine_similarity_matrix(layer_one_hot_df)
        plt.figure(figsize=(16, 8))
        sns.heatmap(cosine_similarity_matrix, annot=True)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "cosine_similarity_matrix.png"))
        plt.close()

class PeftLayerFisherInfoCallback(TrainerCallback):
    
    def __init__(self, output_dir, peft_name, percent=0.7, accumulate_grads=True):
        # working directory
        self.output_dir = output_dir
        
        # middle variables
        self.grads = {}  # Dictionary to store the gradients
        self.temp_params = {}  # Temporary dictionary to store the parameters at the beginning of each step
        self.num_steps = 0
        self.accumulate_grads = accumulate_grads
        
        self.epoch_layer_avg_grads = {}
        self.selected_layer_flags = set()
        self.target_peft_layer_param = {}
        self.peft_name = peft_name
        self.percent = percent

    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize the gradient of each trainable parameter to 0"""
        model = kwargs.get('model', None)
        assert model is not None, "model is None"
        
        peft_layer_names = []
        for name, param in model.named_parameters():
            if param.requires_grad and 'classifier' not in name:
                self.grads[name] = torch.zeros_like(param).to("cpu")  # Initialize the gradient of each trainable parameter to 0
                peft_layer_names.append(name)
        
        for layer_flag in get_taget_models_names(peft_layer_names, self.peft_name):
            self.epoch_layer_avg_grads[layer_flag] = []

    def on_train_end(self, args, state, control, **kwargs):
        """plot fisher info and update the selected layer flags"""
        print("this is on_train_end.")
        layer_fisher_info_df = pd.DataFrame(self.epoch_layer_avg_grads)
        
        # plot the df line fig
        self.plot_line_fig(layer_fisher_info_df)
        
        # plot the df box fig
        self.plot_box_fig(layer_fisher_info_df)
        
        # plot the heatmap using the cos similar bettwen selected layer one hot code in all epoch
        layer_rankd_df = layer_fisher_info_df.rank(axis=1, ascending=False)
        layer_one_hot_df = layer_rankd_df.apply(
            lambda row: row <= int(len(layer_rankd_df.columns) * self.percent), axis=1
        )
        self.plot_heatmap_fig(layer_one_hot_df)
        
        # update the selected layer flags
        last_one_hot = layer_one_hot_df.loc[len(layer_one_hot_df) - 1]
        last_selected_layers = last_one_hot[last_one_hot==1].index.tolist()
        for layer_flag in last_selected_layers:
            self.selected_layer_flags.add(layer_flag)
        
        # save the peft layer param
        model = kwargs.get('model', None)
        for layer_flag in self.selected_layer_flags:
            for peft_layer_name, param in model.named_parameters():
                if layer_flag in peft_layer_name:
                    self.target_peft_layer_param[peft_layer_name] = param.clone().detach()
        print("this is on_train_end")

    def on_epoch_end(self, args, state, control, **kwargs):
        """Calculate the average gradient for each layer"""
        # Calculate the average gradient for each layer
        avg_grads = {}
        for name, param in self.grads.items():
            tmp_param = param.clone().detach().to("cpu")
            avg_grads[name] = tmp_param / self.num_steps
        
        # compute the fisher info of each layer
        layer_fisher_info = {}
        for name, avg_grad in avg_grads.items():
            layer_flag = list(get_taget_models_names([name], self.peft_name))[0]
            layer_fisher_info[layer_flag] = layer_fisher_info.get(layer_flag, 0) + float(torch.mean(avg_grad))
        
        for layer_flag, fisher_info in layer_fisher_info.items():
            self.epoch_layer_avg_grads[layer_flag].append(fisher_info) 

        # save the epoch_layer_avg_grads
        with open(os.path.join(self.output_dir, "epoch_layer_avg_grads.json"), "w") as f:
            json.dump(self.epoch_layer_avg_grads, f, indent=4)
        
        # do not accumulate the grads
        if not self.accumulate_grads:
            for name, param in self.grads.items():
                self.grads[name] = torch.zeros_like(param).to("cpu")
            self.num_steps = 0
        
    def on_step_begin(self, args, state, control, **kwargs):
        """Store the parameters at the beginning of each step"""
        model = kwargs.get('model', None)
        assert model is not None, "model is None"
        for name, param in model.named_parameters():
            if param.requires_grad and 'classifier' not in name:
                self.temp_params[name] = param.clone().detach().to("cpu")

    def on_step_end(self, args, state, control, **kwargs):
        """Calculate the gradient of each trainable parameter and accumulate it"""
        model = kwargs.get('model', None)
        assert model is not None, "model is None"
        for name, param in model.named_parameters():
            if param.requires_grad and 'classifier' not in name:
                tmp_param = param.clone().detach().to("cpu")
                self.grads[name] += (tmp_param - self.temp_params[name]) ** 2
        
        self.num_steps += 1
    
    def plot_line_fig(self, layer_fisher_info_df):
        ax = layer_fisher_info_df.plot(figsize=(16, 8))
        plt.savefig(os.path.join(self.output_dir, "layer_fisher_info_df.png"))
        plt.close()

    def plot_box_fig(self, layer_fisher_info_df):
        ax = layer_fisher_info_df.boxplot(figsize=(16, 8))
        plt.xticks(rotation='vertical')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "layer_fisher_info_df_box.png"))
        plt.close()
    
    def plot_heatmap_fig(self, layer_one_hot_df):
        
        cosine_similarity_matrix = get_cosine_similarity_matrix(layer_one_hot_df)
        plt.figure(figsize=(16, 8))
        sns.heatmap(cosine_similarity_matrix, annot=True)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "cosine_similarity_matrix.png"))
        plt.close()


def get_taget_models_names(selected_peft_layers_names, peft_name):
    """get original models names from selected peft model layers
    example:
        selected_peft_layers_names = [
            'base_model.model.encoder.layer.0.attention.self.query.peft_0.default',
            'base_model.model.encoder.layer.0.attention.self.value.peft_0.default',
            'base_model.model.encoder.layer.0.attention.self.query.peft_0.default',
            'base_model.model.encoder.layer.0.attention.self.value.peft_0.default',
            'base_model.model.encoder.layer.1.attention.self.query.peft_1.default',
            'base_model.model.encoder.layer.1.attention.self.value.peft_1.default',
            'base_model.model.encoder.layer.1.attention.self.query.peft_1.default',
            'base_model.model.encoder.layer.1.attention.self.value.peft_1.default'
        ]
        peft_name = 'peft_0_0'
        target_models_names = ['encoder.layer.0.attention.self.query', 'encoder.layer.0.attention.self.value']
    """
    pattern = rf'base_model\.model\.(.*)\.{peft_name}_.*\.default'
    target_models_names = set()
    for name in selected_peft_layers_names:
        match = re.search(pattern, name)
        if match:
            target_models_names.add(match.group(1))
    return target_models_names

def get_cosine_similarity_matrix(selected_layers_one_hot_df):
    """get the cosine similarity matrix of selected layers one hot code in all epoch
    example:
        selected_layers_one_hot_df = pd.DataFrame([[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1]])
        cosine_similarity_matrix = [[1. 0. 1.]
                                    [0. 1. 0.]
                                    [1. 0. 1.]]
    """
    result_dict = {}
    for epoch in range(1, len(selected_layers_one_hot_df) + 1):
        selected_layer = selected_layers_one_hot_df.loc[epoch-1]
        result_dict[epoch] = np.array(selected_layer)
    
    selected_layer_one_hot_lists = []
    for epoch in range(1, len(result_dict) + 1):
        selected_layer_one_hot_lists.append(result_dict[epoch])
    vectors_matrix = np.array(selected_layer_one_hot_lists)
    cosine_similarity_matrix = cosine_similarity(vectors_matrix)
    return cosine_similarity_matrix

if __name__ == "__main__":
    
    # test get_cosine_similarity_matrix
    selected_layers_one_hot_df = pd.DataFrame([[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1]])
    print(get_cosine_similarity_matrix(selected_layers_one_hot_df))
