from typing import Any, Dict, Union
import torch
from torch import nn
from transformers import Trainer
from transformers.modeling_utils import unwrap_model
# from transformers.utils import is_peft_available
# from peft import PeftModel
from trl import SFTTrainer
from adapters import AdapterTrainer


def dpnoise_post_forward_hook(module, input, output):
    if module.training:
        output_norms = torch.norm(output, p=2, dim=-1) / output.size(-1)
        noise_scale = (output_norms / module.dp_noise_epsilon) * torch.sqrt(2 * torch.log(torch.tensor(1.25 / module.dp_noise_delta)))
        noise_scale = noise_scale ** 2
        noise_scale = noise_scale.unsqueeze(-1).expand_as(output)
        noise = torch.normal(mean=0, std=noise_scale)
        output = output + noise
    return output


class NoiseMixin:
    """
    Add and remove the noise to the input embeddings during one training step.
    """
    def generate_noise_matrix(self, word_embedding):
        # sensetive for each row in the embedding matrix
        row_norms = torch.norm(word_embedding.weight, p=2, dim=1)
        row_norms = row_norms /  word_embedding.weight.shape[1]
        # compute the noise scale for each row
        noise_scale_per_row = (row_norms / self.dp_epsilon) * torch.sqrt(2 * torch.log(torch.tensor(1.25 / self.dp_delta)))
        noise_scale_per_row = noise_scale_per_row ** 2
        noise_scale = noise_scale_per_row.unsqueeze(1).expand_as(word_embedding.weight)
        # generate the noise matrix
        noise_matrix = torch.normal(mean=0, std=noise_scale)
        
        # special token does not have noise
        for special_token_id in self.special_tokens_id:
            noise_matrix[special_token_id, :] = torch.zeros(noise_matrix.shape[1])
        
        return noise_matrix

    def _get_input_embeddings(self, model: nn.Module) -> nn.Embedding:
        unwrapped_model = unwrap_model(model)
        embeddings = unwrapped_model.get_input_embeddings()
        if embeddings is None:
            raise ValueError("The model does not have input embeddings.")
        del unwrapped_model
        return embeddings

    def add_noise(self, model: nn.Module):
        embeddings = self._get_input_embeddings(model)
        self.noise_matrix = self.generate_noise_matrix(embeddings)
        embeddings.weight.data.add_(self.noise_matrix)

    def remove_noise(self, model: nn.Module):
        embeddings = self._get_input_embeddings(model)
        embeddings.weight.data.sub_(self.noise_matrix)
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        self.add_noise(model)
        result = super().training_step(model, inputs)
        self.remove_noise(model)
        return result


class NoiseTrainer(NoiseMixin, Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dp_epsilon = 0.1
        self.dp_delta = 1e-5
        self.special_tokens_id = [
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id
        ]

class NoiseAdapterTrainer(NoiseMixin, AdapterTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dp_epsilon = 0.1
        self.dp_delta = 1e-5
        self.special_tokens_id = [
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id
        ]


class NoiseMixin2:
    """
    Add and remove the noise to the input embeddings during each embedding.forward()
    """
    def train(self, resume_from_checkpoint = None, trial = None, ignore_keys_for_eval = None, **kwargs):
        self.model = self._activate_dpnoise(self.model)
        result = super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
        self._deactivate_dpnoise(self.model)
        return result

    def _get_input_embeddings(self, model: nn.Module) -> nn.Embedding:
        unwrapped_model = unwrap_model(model)
        embeddings = unwrapped_model.get_input_embeddings()
        if embeddings is None:
            raise ValueError("The model does not have input embeddings.")
        del unwrapped_model
        return embeddings

    def _activate_dpnoise(self, model: nn.Module) -> nn.Module:
        embeddings = self._get_input_embeddings(model)
        embeddings.dp_noise_epsilon = self.dp_epsilon
        embeddings.dp_noise_delta = self.dp_delta
        self.dpnoise_hook_handle = embeddings.register_forward_hook(dpnoise_post_forward_hook)
        return model

    def _deactivate_dpnoise(self, model: nn.Module):
        embeddings = self._get_input_embeddings(model)
        self.dpnoise_hook_handle.remove()
        del embeddings.dp_noise_epsilon, embeddings.dp_noise_delta


class NoiseTrainer2(NoiseMixin2, Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dp_epsilon = 0.1
        self.dp_delta = 1e-5


class NoiseAdapterTrainer2(NoiseMixin2, AdapterTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dp_epsilon = 0.1
        self.dp_delta = 1e-5


class DPNoiseAdapterTrainerWithNLG(AdapterTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dp_epsilon = 0.1
        self.dp_delta = 1e-5
    
    def generate_noise_matrix(self, word_embedding):
        # sensetive for each row in the embedding matrix
        row_norms = torch.norm(word_embedding.weight, p=2, dim=1)
        row_norms = row_norms /  word_embedding.weight.shape[1]
        # compute the noise scale for each row
        noise_scale_per_row = (row_norms / self.dp_epsilon) * torch.sqrt(2 * torch.log(torch.tensor(1.25 / self.dp_delta)))
        noise_scale_per_row = noise_scale_per_row ** 2
        noise_scale = noise_scale_per_row.unsqueeze(1).expand_as(word_embedding.weight)
        # generate the noise matrix
        noise_matrix = torch.normal(mean=0, std=noise_scale)
        
        # special token does not have noise
        eos_token_id = self.tokenizer.eos_token_id
        noise_matrix[eos_token_id, :] = torch.zeros(noise_matrix.shape[1])
        
        return noise_matrix

    def add_noise(self, model: nn.Module):
        unwrapped_model = unwrap_model(model)
        embeddings = unwrapped_model.get_input_embeddings()
        del unwrapped_model
        
        self.noise_matrix = self.generate_noise_matrix(embeddings)
        embeddings.weight.data.add_(self.noise_matrix)

    def remove_noise(self, model: nn.Module):
        unwrapped_model = unwrap_model(model)
        embeddings = unwrapped_model.get_input_embeddings()
        del unwrapped_model
        
        embeddings.weight.data.sub_(self.noise_matrix)
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        
        # add noise to Embedding layer
        self.add_noise(model)
        
        result = super().training_step(model, inputs)
        
        # remove noise from Embedding layer
        self.remove_noise(model)
        
        return result

class DPNoiseAdapterTrainerWithNLU(AdapterTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dp_epsilon = 0.1
        self.dp_delta = 1e-5
    
    def generate_noise_matrix(self, word_embedding):
        # sensetive for each row in the embedding matrix
        row_norms = torch.norm(word_embedding.weight, p=2, dim=1)
        row_norms = row_norms /  word_embedding.weight.shape[1]
        # compute the noise scale for each row
        noise_scale_per_row = (row_norms / self.dp_epsilon) * torch.sqrt(2 * torch.log(torch.tensor(1.25 / self.dp_delta)))
        noise_scale_per_row = noise_scale_per_row ** 2
        noise_scale = noise_scale_per_row.unsqueeze(1).expand_as(word_embedding.weight)
        # generate the noise matrix
        noise_matrix = torch.normal(mean=0, std=noise_scale)
        return noise_matrix

    def add_noise(self, model: nn.Module):
        unwrapped_model = unwrap_model(model)
        embeddings = unwrapped_model.get_input_embeddings()
        del unwrapped_model
        
        self.noise_matrix = self.generate_noise_matrix(embeddings)
        embeddings.weight.data.add_(self.noise_matrix)

    def remove_noise(self, model: nn.Module):
        unwrapped_model = unwrap_model(model)
        embeddings = unwrapped_model.get_input_embeddings()
        del unwrapped_model
        
        embeddings.weight.data.sub_(self.noise_matrix)
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:

        # add noise to Embedding layer
        self.add_noise(model)
        
        result = super().training_step(model, inputs)
        
        # remove noise from Embedding layer
        self.remove_noise(model)
        
        return result


class DPNoiseTrainer(NoiseTrainer):
    def __init__(self, dp_noise, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dp_epsilon = getattr(dp_noise, "dp_epsilon", 0.1)
        self.dp_delta = getattr(dp_noise, "dp_delta", 1e-5)
    
    def generate_noise_matrix(self, word_embedding):
        # sensetive for each row in the embedding matrix
        row_norms = torch.norm(word_embedding.weight, p=2, dim=1)
        row_norms = row_norms /  word_embedding.weight.shape[1]
        # compute the noise scale for each row
        noise_scale_per_row = (row_norms / self.dp_epsilon) * torch.sqrt(2 * torch.log(torch.tensor(1.25 / self.dp_delta)))
        noise_scale_per_row = noise_scale_per_row ** 2
        noise_scale = noise_scale_per_row.unsqueeze(1).expand_as(word_embedding.weight)
        # generate the noise matrix
        noise_matrix = torch.normal(mean=0, std=noise_scale)
        return noise_matrix
    
    def add_noise(self, model: nn.Module):
        
        unwrapped_model = unwrap_model(model)
        embeddings = unwrapped_model.get_input_embeddings()
        del unwrapped_model
        
        self.noise_matrix = self.generate_noise_matrix(embeddings)
        # print("noise matrix: ", self.noise_matrix[0][:5])
        # print("word embedding: ", embeddings.weight[0][:5])
        embeddings.weight.data.add_(self.noise_matrix)
        # print("word embedding after add noise: ", embeddings.weight[0][:5])
    
    def remove_noise(self, model: nn.Module):
        
        unwrapped_model = unwrap_model(model)
        embeddings = unwrapped_model.get_input_embeddings()
        del unwrapped_model
        
        embeddings.weight.data.sub_(self.noise_matrix)

class DPNoiseSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dp_epsilon = getattr(self.args, "dp_epsilon", 0.1)
        self.dp_delta = getattr(self.args, "dp_delta", 1e-5)
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        
        # add noise to Embedding layer
        self.add_noise(model)
        
        result = super().training_step(model, inputs)
        
        # remove noise from Embedding layer
        self.remove_noise(model)
        
        return result
    
    def generate_noise_matrix(self, word_embedding):
        # sensetive for each row in the embedding matrix
        row_norms = torch.norm(word_embedding.weight, p=2, dim=1)
        row_norms = row_norms /  word_embedding.weight.shape[1]
        # compute the noise scale for each row
        noise_scale_per_row = (row_norms / self.dp_epsilon) * torch.sqrt(2 * torch.log(torch.tensor(1.25 / self.dp_delta)))
        noise_scale_per_row = noise_scale_per_row ** 2
        noise_scale = noise_scale_per_row.unsqueeze(1).expand_as(word_embedding.weight)
        # generate the noise matrix
        noise_matrix = torch.normal(mean=0, std=noise_scale)
        return noise_matrix
    
    def add_noise(self, model: nn.Module):
        
        unwrapped_model = unwrap_model(model)
        embeddings = unwrapped_model.get_input_embeddings()
        del unwrapped_model
        
        self.noise_matrix = self.generate_noise_matrix(embeddings)
        # print("noise matrix: ", self.noise_matrix[0][:5])
        # print("word embedding: ", embeddings.weight[0][:5])
        embeddings.weight.data.add_(self.noise_matrix)
        # print("word embedding after add noise: ", embeddings.weight[0][:5])
    
    def remove_noise(self, model: nn.Module):
        
        unwrapped_model = unwrap_model(model)
        embeddings = unwrapped_model.get_input_embeddings()
        del unwrapped_model
        
        embeddings.weight.data.sub_(self.noise_matrix)
    

class DPNoiseTrainer2(NoiseTrainer):
    def __init__(self, dp_noise, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dp_epsilon = getattr(dp_noise, "dp_epsilon", 0.1)
        self.dp_delta = getattr(dp_noise, "dp_delta", 1e-5)
    
    def train(self, resume_from_checkpoint = None, trial = None, ignore_keys_for_eval = None, **kwargs,
    ):
        self.model = self._activate_dpnoise(self.model)
        result = super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
        self._deactivate_dpnoise(self.model)
        return result
    
    def _activate_dpnoise(self, model):
        unwrapped_model = unwrap_model(model)

        # if is_peft_available() and isinstance(unwrapped_model, PeftModel):
        #     embeddings = unwrapped_model.base_model.model.get_input_embeddings()
        # else:
        #     embeddings = unwrapped_model.get_input_embeddings()
        embeddings = unwrapped_model.get_input_embeddings()

        del unwrapped_model

        embeddings.dp_noise_epsilon = self.dp_epsilon
        embeddings.dp_noise_delta = self.dp_delta
        hook_handle = embeddings.register_forward_hook(dpnoise_post_forward_hook)
        self.dpnoise_hook_handle = hook_handle
        return model
    
    def _deactivate_dpnoise(self, model):
        unwrapped_model = unwrap_model(model)

        # if is_peft_available() and isinstance(unwrapped_model, PeftModel):
        #     embeddings = unwrapped_model.base_model.model.get_input_embeddings()
        # else:
        #     embeddings = unwrapped_model.get_input_embeddings()
        embeddings = unwrapped_model.get_input_embeddings()

        self.dpnoise_hook_handle.remove()
        del unwrapped_model, embeddings.dp_noise_epsilon, embeddings.dp_noise_delta

