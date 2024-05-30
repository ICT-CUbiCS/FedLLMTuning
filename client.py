import os
import json
import torch
from dataclasses import field
from typing import Union, Optional, Dict

from datasets import load_from_disk
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback
from peft import PeftModel
from trl import SFTTrainer
from adapters import AdapterTrainer

from torch import nn
from torch.utils.data import Dataset
from transformers import PreTrainedModel

from callbacks import PeftLayerFisherInfoCallback, AdapterLayerFisherInfoCallback
from config import glue_task_metrics
from noise_trainer import DPNoiseTrainer, DPNoiseSFTTrainer

from noise_trainer import NoiseTrainer, NoiseAdapterTrainer

class Client:
    
    client_id: int = field(
        default=0, metadata={"help": "The client id"}
    )
    model: Union[PreTrainedModel, nn.Module] = field(
        default=None, metadata={"help": "The model"}
    )
    local_output_dir: str = field(
        metadata={"help": "The local output directory"}
    )
    final_model_dir: str = field(
        metadata={"help": "The final model directory"}
    )
    local_trainer: Trainer = field(
        default=None, metadata={"help": "The local trainer"}
    )
    data_name: str = field(
        default=None, metadata={"help": "The data name"}
    )
    local_train_dataset: Optional[Dataset] = field(
        default=None, metadata={"help": "The local train dataset"}
    )
    local_eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = field(
        default=None, metadata={"help": "The local eval dataset"}
    )
    
    def __init__(self, **kwargs) -> None:
        self.client_id = kwargs.get("client_id")
        self.model = kwargs.get("model")
        output_dir = kwargs.get("output_dir")
        self.local_output_dir = os.path.join(output_dir, "client_{}".format(self.client_id))
        self.final_model_dir = os.path.join(self.local_output_dir, "final_model")
        
        self.data_name = kwargs.get("data_name", None)
    
    def load_local_data(self, train_dataset, test_dataset):
        self.local_train_dataset = train_dataset
        self.local_eval_dataset = test_dataset
    
    def load_local_trainer(self, trainer_cls, **kwargs):
        tokenizer, compute_metrics, train_args, callbacks = self._prepare_trainer_args(**kwargs)
        self.local_trainer = trainer_cls(
            model=self.model, args=train_args,
            train_dataset=self.local_train_dataset, eval_dataset=self.local_eval_dataset,
            tokenizer=tokenizer, compute_metrics=compute_metrics,
            callbacks=callbacks
        )
    
    def _prepare_trainer_args(self, **kwargs):
        tokenizer = kwargs.get("tokenizer")
        compute_metrics = kwargs.get("compute_metrics", None)
        
        # training arguments setting
        train_args_dict = kwargs.get("train_args_dict")
        train_args_dict.update({
            "logging_dir": kwargs.get("logging_dir"),
        })
        train_args_dict, callbacks = self._set_train_args(train_args_dict)
        train_args = TrainingArguments(**train_args_dict)
        
        return tokenizer, compute_metrics, train_args, callbacks
    
    def _set_train_args(self, train_args_dict):
        
        callbacks = []
        train_args_dict.update({
            "output_dir": self.local_output_dir,
        })
        
        if self.data_name is not None:
            train_args_dict.update({
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss"
            })
            if self.data_name and self.data_name in glue_task_metrics.keys():
                train_args_dict['metric_for_best_model'] = glue_task_metrics[self.data_name][0]
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=5))
        
        return train_args_dict, callbacks

    def train(self):
        print("training...")
        resume_from_checkpoint = False
        for dir_name in os.listdir(self.local_output_dir):
            if os.path.isdir(os.path.join(self.local_output_dir, dir_name)) and "checkpoint" in dir_name:
                resume_from_checkpoint = True
                break
        self.local_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        self.local_trainer.save_model(self.final_model_dir)
    
    
    def evaluate(self):
        print("evaluating...")
        eval_metrics = self.local_trainer.evaluate()
        print("eval_metrics: ", eval_metrics)
        with open(os.path.join(self.final_model_dir, "eval_metrics.json"), "w") as f:
            json.dump(eval_metrics, f, indent=4)
        return eval_metrics

    def select_target_layers(self, trainer_cls, callback_cls, **kwargs):
        train_args_dict = kwargs.get("train_args_dict")
        train_args_dict.update({
            "output_dir": os.path.join(self.local_output_dir, "peft_layer_selection"),
            
            # early stop args
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            
            "num_train_epochs": 5
        })
        if self.data_name is not None and self.data_name in glue_task_metrics.keys():
            train_args_dict['metric_for_best_model'] = glue_task_metrics[self.data_name][0]
        
        percent = kwargs.get("percent")
        fisherinfo_callback = callback_cls(
            train_args_dict["output_dir"], kwargs.get("peft_name").lower(),
            percent=percent, accumulate_grads=True
        )
        train_args = TrainingArguments(**train_args_dict)
        tokenizer = kwargs.get("tokenizer")
        compute_metrics = kwargs.get("compute_metrics")
        trainer = trainer_cls(
            model=self.model, args=train_args,
            train_dataset=self.local_train_dataset, eval_dataset=self.local_eval_dataset,
            tokenizer=tokenizer, compute_metrics=compute_metrics,
            callbacks=[fisherinfo_callback]
        )
        trainer.train()
        
        return fisherinfo_callback


class ClientWithTrainer(Client):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def load_local_trainer(self, **kwargs):
        add_noise = kwargs.get("add_noise")
        if add_noise:
            super().load_local_trainer(NoiseTrainer, **kwargs)
            self.local_trainer.dp_noise = kwargs.get("dp_noise")
        else:
            super().load_local_trainer(Trainer, **kwargs)
    
    def select_target_layers(self, **kwargs):
        fisherinfo_callback = super().select_target_layers(Trainer, PeftLayerFisherInfoCallback, **kwargs)
        
        selected_layer_flags = fisherinfo_callback.selected_layer_flags
        assert len(selected_layer_flags) > 0, "selected_layer_flags is empty"
        target_peft_layer_param = fisherinfo_callback.target_peft_layer_param
        return selected_layer_flags, target_peft_layer_param


class ClientWithAdapterTrainer(Client):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def load_local_trainer(self, **kwargs):
        add_noise = kwargs.get("add_noise")
        if add_noise:
            super().load_local_trainer(NoiseAdapterTrainer, **kwargs)
            self.local_trainer.dp_noise = kwargs.get("dp_noise")
        else:
            super().load_local_trainer(AdapterTrainer, **kwargs)
    
    def select_target_layers(self, **kwargs):
        fisherinfo_callback = super().select_target_layers(AdapterTrainer, AdapterLayerFisherInfoCallback, **kwargs)
        
        unselected_layer_index = fisherinfo_callback.unselected_layers
        assert len(unselected_layer_index) > 0, "unselected_layer_index is empty"
        target_adapter_param = fisherinfo_callback.target_adapter_layer_param
        return unselected_layer_index, target_adapter_param


class NLUClientFLTrainer():
    def __init__(self, client_id, model, data_dir, output_dir, fl_dir) -> None:
        
        self.client_id = client_id
        self.model = model
    
        self.train_data_dir = os.path.join(data_dir, f"train_dataset_{client_id}")
        self.eval_data_dir = os.path.join(data_dir, f"eval_dataset_{client_id}")
        
        self.output_dir = output_dir
        self.local_output_dir = os.path.join(output_dir, f"client_{client_id}")
        self.fl_dir = fl_dir
    
    def preprare_local_dataset(self):
        self.train_dataset = load_from_disk(self.train_data_dir)
        self.eval_dataset = load_from_disk(self.eval_data_dir)
    
    def build_local_trainer(self, tokenizer, compute_metrics, add_noise, **train_kwargs):
        
        train_kwargs.update({
            "output_dir": self.local_output_dir,
        })
        
        training_args = TrainingArguments(**train_kwargs)
        if not add_noise:
            self.local_trainer = Trainer(
                model=self.model,
                train_dataset=self.train_dataset, eval_dataset=self.eval_dataset,
                args=training_args,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics
            )
        else:
            tmp_noise = {"dp_epsilon": 0.1, "dp_delta": 1e-5}
            self.local_trainer = DPNoiseTrainer(
                dp_noise=tmp_noise,
                model=self.model,
                train_dataset=self.train_dataset, eval_dataset=self.eval_dataset,
                args=training_args,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics
            )
    
    def initiate_local_training(self):
        self.model.config.use_cache = False
    
    def train(self):
        self.local_trainer.train()
    
    def terminate_local_training(self, trainable_param_old, epoch):
        
        # save the trainable weight
        trainable_param_new = self._get_trainable_param()
        epoch_saved_dir = os.path.join(self.fl_dir, str(epoch), f"local_output_{self.client_id}")
        os.makedirs(epoch_saved_dir, exist_ok=True)
        torch.save(trainable_param_new, os.path.join(epoch_saved_dir, "pytorch_model.bin"))
        
        # reduct global model
        for name, param in self.model.named_parameters():
            old_param = trainable_param_old.get(name, None)
            if old_param is not None:
                param.data.copy_(old_param.clone())
        return self.model
    
    def _get_trainable_param(self):
        param_dict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_dict[name] = param.clone().detach()
        return param_dict
    
    def select_peft_target_layers(self, tokenizer, percent, compute_metrics=None, **kwargs):
        
        assert isinstance(self.model, PeftModel), "model is not PeftModel"
        kwargs.update({
            "output_dir": os.path.join(self.local_output_dir, "peft_layer_selection"),
            
            # early stop args
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            
            "num_train_epochs": 5
        })
        
        peft_layer_fisherinfo_callback = PeftLayerFisherInfoCallback(
            kwargs["output_dir"],
            peft_name=self.model.peft_config['default'].peft_type.value.lower(),
            percent=percent, accumulate_grads=True
        )
        
        training_args = TrainingArguments(**kwargs)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[peft_layer_fisherinfo_callback]
        )
        trainer.train()
        epoch_layer_avg_grads = peft_layer_fisherinfo_callback.epoch_layer_avg_grads
        last_epoch_avg_grads = {
            layer: grads[-1] for layer, grads in epoch_layer_avg_grads.items()
        }
        peft_layer_param = {}
        for layer_flag in last_epoch_avg_grads.keys():
            for name, param in self.model.named_parameters():
                if param.requires_grad and layer_flag in name:
                    peft_layer_param[name] = param.clone().detach()
        
        return last_epoch_avg_grads, peft_layer_param
        

class NLGClientFLTrainer():
    def __init__(self, client_id, model, data_dir, output_dir, fl_dir) -> None:
        
        self.client_id = client_id
        self.model = model
    
        self.train_data_dir = os.path.join(data_dir, f"train_dataset_{client_id}")
        self.eval_data_dir = os.path.join(data_dir, f"eval_dataset_{client_id}")
        
        self.output_dir = output_dir
        self.local_output_dir = os.path.join(output_dir, f"client_{client_id}")
        self.fl_dir = fl_dir
    
    def preprare_local_dataset(self):
        self.train_dataset = load_from_disk(self.train_data_dir)
        self.eval_dataset = load_from_disk(self.eval_data_dir)
    
    def build_local_trainer(self, tokenizer, add_noise, **train_kwargs):
        train_kwargs.update({
            "output_dir": self.local_output_dir,
            
            # do not evaluate during training
            "evaluation_strategy": "no"
        })
        
        training_args = TrainingArguments(**train_kwargs)
        if not add_noise:
            self.local_trainer = Trainer(
                model=self.model,
                train_dataset=self.train_dataset, eval_dataset=self.eval_dataset,
                args=training_args,
                tokenizer=tokenizer
            )
        else:
            tmp_noise = {"dp_epsilon": 0.15, "dp_delta": 1e-5}
            self.local_trainer = DPNoiseTrainer(
                dp_noise=tmp_noise,
                model=self.model,
                train_dataset=self.train_dataset, eval_dataset=self.eval_dataset,
                args=training_args,
                tokenizer=tokenizer
            )
    
    def initiate_local_training(self):
        self.model.config.use_cache = False
    
    def train(self):
        self.local_trainer.train()
    
    def terminate_local_training(self, trainable_param_old, epoch):
        # save the trainable weight
        trainable_param_new = self._get_trainable_param()
        epoch_saved_dir = os.path.join(self.fl_dir, str(epoch), f"local_output_{self.client_id}")
        os.makedirs(epoch_saved_dir, exist_ok=True)
        torch.save(trainable_param_new, os.path.join(epoch_saved_dir, "pytorch_model.bin"))
        
        # reduct global model
        for name, param in self.model.named_parameters():
            old_param = trainable_param_old.get(name, None)
            if old_param is not None:
                param.data.copy_(old_param.clone())
        return self.model
    
    def _get_trainable_param(self):
        param_dict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_dict[name] = param.clone().detach()
        return param_dict

    def select_peft_target_layers(self, tokenizer, percent, **kwargs):
        
        assert isinstance(self.model, PeftModel), "model is not PeftModel"
        kwargs.update({
            "output_dir": os.path.join(self.local_output_dir, "peft_layer_selection"),
            
            # early stop args
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            
            "num_train_epochs": 5
        })
        peft_layer_fisherinfo_callback = PeftLayerFisherInfoCallback(
            kwargs["output_dir"],
            peft_name=self.model.peft_config['default'].peft_type.value.lower(),
            percent=percent, accumulate_grads=True
        )
        training_args = TrainingArguments(**kwargs)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=tokenizer,
            callbacks=[peft_layer_fisherinfo_callback]
        )
        trainer.train()
        epoch_layer_avg_grads = peft_layer_fisherinfo_callback.epoch_layer_avg_grads
        last_epoch_avg_grads = {
            layer: grads[-1] for layer, grads in epoch_layer_avg_grads.items()
        }
        peft_layer_param = {}
        for layer_flag in last_epoch_avg_grads.keys():
            for name, param in self.model.named_parameters():
                if param.requires_grad and layer_flag in name:
                    peft_layer_param[name] = param.clone().detach()
        
        return last_epoch_avg_grads, peft_layer_param
        


class NLGClientWithTrainer():
    def __init__(self, client_id, model, output_dir, logs_dir=None) -> None:
        self.client_id = client_id
        self.model = model
        self.local_output_dir = os.path.join(output_dir, "client_{}".format(self.client_id))
        self.logging_dir = logs_dir if logs_dir is not None else os.path.join(self.local_output_dir, "logs")
        self.final_model_dir = os.path.join(self.local_output_dir, "final_model")
    
    def load_local_data(self, train_dataset, test_dataset):
        self.local_train_dataset = train_dataset
        self.local_eval_dataset = test_dataset
    
    def load_local_trainer(self, tokenizer, add_noise=False, **kwargs):
        
        assert isinstance(self.model, PeftModel), "model is not PeftModel"
        # train args
        kwargs.update({
            "output_dir": self.local_output_dir,
            "logging_dir": self.logging_dir,
        })
        training_args = TrainingArguments(**kwargs)
        
        if not add_noise:
            self.local_trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.local_train_dataset,
                eval_dataset=self.local_eval_dataset,
                tokenizer=tokenizer,
            )
        else:
            tmp_noise = {"dp_epsilon": 0.15, "dp_delta": 1e-5}
            self.local_trainer = DPNoiseTrainer(
                dp_noise=tmp_noise,
                model=self.model,
                args=training_args,
                train_dataset=self.local_train_dataset,
                eval_dataset=self.local_eval_dataset,
                tokenizer=tokenizer,
            )
    
    def train(self):
        resume_from_checkpoint = False
        for dir_name in os.listdir(self.local_output_dir):
            if os.path.isdir(os.path.join(self.local_output_dir, dir_name)) and "checkpoint" in dir_name:
                resume_from_checkpoint = True
                break
        self.local_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        self.local_trainer.save_model(self.final_model_dir)

    def selecte_peft_target_layers(self, tokenizer, percent, **kwargs):
        
        assert isinstance(self.model, PeftModel), "model is not PeftModel"
        
        # training arguments setting
        kwargs.update({
            "output_dir": os.path.join(self.local_output_dir, "peft_layer_selection"),
            "logging_dir": os.path.join(self.local_output_dir, "peft_layer_selection", "logs"),
            
            # "num_train_epochs": 1 # test
            "num_train_epochs": 5
        })
        
        peft_layer_fisherinfo_callback = PeftLayerFisherInfoCallback(
            kwargs["output_dir"], 
            peft_name=self.model.peft_config['default'].peft_type.value.lower(),
            percent=percent, accumulate_grads=True
        )
        
        callbacks = [
            peft_layer_fisherinfo_callback
        ]
        
        training_args = TrainingArguments(**kwargs)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.local_train_dataset.shuffle(42).select(range(int(0.1*len(self.local_train_dataset)))),
            eval_dataset=self.local_eval_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks
        )
        trainer.train()
        
        selected_layer_flags = peft_layer_fisherinfo_callback.selected_layer_flags
        assert len(selected_layer_flags) > 0, "selected_layer_flags is empty"
        target_peft_layer_param = peft_layer_fisherinfo_callback.target_peft_layer_param
        return selected_layer_flags, target_peft_layer_param


class NLGClientWithSFTTrainer():
    def __init__(self, client_id, model, output_dir) -> None:
        self.client_id = client_id
        self.model = model
        self.local_output_dir = os.path.join(output_dir, "client_{}".format(self.client_id))
        self.logging_dir = os.path.join(self.local_output_dir, "logs")
        self.final_model_dir = os.path.join(self.local_output_dir, "final_model")
    
    def load_local_data(self, train_dataset, test_dataset):
        self.local_train_dataset = train_dataset
        self.local_eval_dataset = test_dataset
    
    def load_local_trainer(self, tokenizer, format_func, add_noise=False, **kwargs):
        
        assert isinstance(self.model, PeftModel), "model is not PeftModel"
        # train args
        kwargs.update({
            "output_dir": self.local_output_dir,
            "logging_dir": self.logging_dir,
        })
        training_args = TrainingArguments(**kwargs)
        
        # noise
        trainer_cls = SFTTrainer if add_noise is False else DPNoiseSFTTrainer
        
        self.local_trainer = trainer_cls(
            model=self.model,
            train_dataset=self.local_train_dataset,
            eval_dataset=self.local_eval_dataset,
            max_seq_length=512,
            tokenizer=tokenizer, packing=True,
            formatting_func=format_func,
            args=training_args
        )
    
    def train(self):
        resume_from_checkpoint = False
        # for dir_name in os.listdir(self.local_output_dir):
        #     if os.path.isdir(os.path.join(self.local_output_dir, dir_name)) and "checkpoint" in dir_name:
        #         resume_from_checkpoint = True
        #         break
        self.local_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        self.local_trainer.save_model(self.final_model_dir)
    
    def selecte_peft_target_layers(self, tokenizer, percent, format_func, **kwargs):
        
        assert isinstance(self.model, PeftModel), "model is not PeftModel"
        
        # training arguments setting
        kwargs.update({
            "output_dir": os.path.join(self.local_output_dir, "peft_layer_selection"),
            "logging_dir": os.path.join(self.local_output_dir, "peft_layer_selection", "logs"),
            
            # "num_train_epochs": 1 # test
            # "num_train_epochs": 1
        })
        
        peft_layer_fisherinfo_callback = PeftLayerFisherInfoCallback(
            kwargs["output_dir"], 
            peft_name=self.model.peft_config['default'].peft_type.value.lower(),
            percent=percent, accumulate_grads=True
        )
        
        callbacks = [
            peft_layer_fisherinfo_callback
        ]
        
        training_args = TrainingArguments(**kwargs)
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.local_train_dataset.select(range(int(0.1*len(self.local_train_dataset)))),
            max_seq_length=512,
            tokenizer=tokenizer, packing=True,
            formatting_func=format_func,
            args=training_args,
            callbacks=callbacks
        )
        trainer.train()
        
        selected_layer_flags = peft_layer_fisherinfo_callback.selected_layer_flags
        assert len(selected_layer_flags) > 0, "selected_layer_flags is empty"
        target_peft_layer_param = peft_layer_fisherinfo_callback.target_peft_layer_param
        return selected_layer_flags, target_peft_layer_param
