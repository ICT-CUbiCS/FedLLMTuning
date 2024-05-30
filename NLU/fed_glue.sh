#!/bin/bash

script_name="python fed_glue.py"

# model_names=("roberta-base")
model_names=("bert-base-uncased")

peft_names=("lora")

noise_types=("" 1)

data_names=("cola" "mrpc" "rte" "sst2" "stsb")

percent_values=(0.5 0.6 0.7 "")

for data_name in "${data_names[@]}"; do
  for model_name in "${model_names[@]}"; do
    for peft_name in "${peft_names[@]}"; do
      for noise_type in "${noise_types[@]}"; do
        for percent in "${percent_values[@]}"; do
          cmd="$script_name --model_name $model_name --peft_name $peft_name --data_name $data_name"
          if [ -n "$percent" ]; then
            cmd="$cmd --percent $percent"
          fi
          if [ -n "$noise_type" ]; then
            cmd="$cmd --add_noise $noise_type"
          fi
          echo $cmd
          # eval $cmd
        done
      done
    done
  done
done