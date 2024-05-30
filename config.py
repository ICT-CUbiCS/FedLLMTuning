import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Local path
#MODEL_DIR = os.path.join("/home/shared/huggingface/LargeModel")
#DATA_DIR = os.path.join("/home/shared/huggingface/Datasets")
# SoThisAI path
MODEL_DIR = os.path.join("/public/zenghui/LargeModel")
DATA_DIR = os.path.join("/public/zenghui/Datasets")
# AutoDL path
# MODEL_DIR = os.path.join("/root/autodl-tmp/LargeModel")
# DATA_DIR = os.path.join("/root/autodl-tmp/Datasets")

# metrics
evaluates_metric_dir = {
    "glue": os.path.join(PROJECT_DIR, "metrics", "glue"),
    "nlg": os.path.join(PROJECT_DIR, "metrics", "nlg"),
    "accuracy": os.path.join(PROJECT_DIR, "metrics", "accuracy"),
}

# peft
PEFT_NAMES = ['lora', 'IA3', 'bottleneck']
adapter_peft_names = ['bottleneck']

# noise
DP_NOISE = [
    None, 
    {"dp_noise": {"dp_epsilon": 0.1, "dp_delta": 1e-5}}, 
    {"dp_noise2": {"dp_epsilon": 0.1, "dp_delta": 1e-5}},
]

# common
metrics_name_full_to_short = {
    "matthews_correlation": "mcc",
    "accuracy": "acc",
    "f1": "f1",
    "pearson": "pearson",
    "spearmanr": "spearmanr",
}
model_name_full_to_short = {
    "bert-base-uncased": "bert",
    "roberta-base": "roberta",
    "gpt2-medium": "GPT2-M",
    "gpt2-large": "GPT2-L",
    "llama2-7b": "LLaMA2",
    "LLMPruner": "LLMPruner",
    "ChildTuning": "ChildTuning",
    "Sheared-LLaMA-2.7B": "Sheared-LLaMA-2.7B",
    "Sheared-LLaMA-1.3B": "Sheared-LLaMA-1.3B"
}

# NLU
NLU_DIR = os.path.join(PROJECT_DIR, "NLU")
nlu_model_names = ['bert-base-uncased', 'roberta-base']
glue_task_input_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    # "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    # "wnli": ("sentence1", "sentence2"),
}
glue_task_num_labels = {
    "mnli": 3,
    "cola": 2,
    "mrpc": 2,
    # "qnli": 2,
    "qqp": 2,
    "rte": 2,
    "sst2": 2,
    "stsb": 1,
    # "wnli": 2,
}

# NLU evaluate
glue_task_metrics = {
    "cola": ["matthews_correlation"],
    "mnli": ["accuracy"],
    "mrpc": ["accuracy", "f1"],
    # "qnli": ["accuracy"],
    "qqp": ["accuracy", "f1"],
    "rte": ["accuracy"],
    "sst2": ["accuracy"],
    "stsb": ["pearson", "spearmanr"],
    # "wnli": ["accuracy"],
}

# NLG
NLG_DIR = os.path.join(PROJECT_DIR, "NLG")
nlg_dataset_to_keys = {
    "e2e_nlg": ("meaning_representation", "human_reference"),
}
nlg_model_names = ['gpt2-medium', 'gpt2-large', 'llama2-7b', 'Sheared-LLaMA-2.7B', 'Sheared-LLaMA-1.3B']
nlg_eval_metrics_keys = {
    "bleu": ["bleu"],
    "rouge": ["rouge1", "rouge2", "rougeL"],
    "meteor": ["meteor"],
    "nist_mt": ["nist_mt"],
}
