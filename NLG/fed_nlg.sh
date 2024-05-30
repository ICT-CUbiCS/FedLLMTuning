#!/bin/bash

script_name="python fed_nlg.py"

model_names=('gpt2-medium' 'gpt2-large' 'llama2-7b')

peft_names=("lora")

noise_types=("" 1)

data_names=("e2e_nlg")

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


# python fed_nlg.py --model_name gpt2-medium --peft_name lora --data_name e2e_nlg --percent 0.5
# python fed_nlg.py --model_name gpt2-medium --peft_name lora --data_name e2e_nlg --percent 0.6
# python fed_nlg.py --model_name gpt2-medium --peft_name lora --data_name e2e_nlg --percent 0.7
# python fed_nlg.py --model_name gpt2-medium --peft_name lora --data_name e2e_nlg

# python fed_nlg.py --model_name gpt2-medium --peft_name lora --data_name e2e_nlg --percent 0.5 --add_noise 1
# python fed_nlg.py --model_name gpt2-medium --peft_name lora --data_name e2e_nlg --percent 0.6 --add_noise 1
# python fed_nlg.py --model_name gpt2-medium --peft_name lora --data_name e2e_nlg --percent 0.7 --add_noise 1
# python fed_nlg.py --model_name gpt2-medium --peft_name lora --data_name e2e_nlg --add_noise 1

# python fed_nlg.py --model_name gpt2-large --peft_name lora --data_name e2e_nlg --percent 0.5
# python fed_nlg.py --model_name gpt2-large --peft_name lora --data_name e2e_nlg --percent 0.6
# python fed_nlg.py --model_name gpt2-large --peft_name lora --data_name e2e_nlg --percent 0.7
# python fed_nlg.py --model_name gpt2-large --peft_name lora --data_name e2e_nlg

# python fed_nlg.py --model_name gpt2-large --peft_name lora --data_name e2e_nlg --percent 0.5 --add_noise 1
# python fed_nlg.py --model_name gpt2-large --peft_name lora --data_name e2e_nlg --percent 0.6 --add_noise 1
# python fed_nlg.py --model_name gpt2-large --peft_name lora --data_name e2e_nlg --percent 0.7 --add_noise 1
# python fed_nlg.py --model_name gpt2-large --peft_name lora --data_name e2e_nlg --add_noise 1

# python fed_nlg.py --model_name llama2-7b --peft_name lora --data_name e2e_nlg --percent 0.5
# python fed_nlg.py --model_name llama2-7b --peft_name lora --data_name e2e_nlg --percent 0.6
# python fed_nlg.py --model_name llama2-7b --peft_name lora --data_name e2e_nlg --percent 0.7
# python fed_nlg.py --model_name llama2-7b --peft_name lora --data_name e2e_nlg

# python fed_nlg.py --model_name llama2-7b --peft_name lora --data_name e2e_nlg --percent 0.5 --add_noise 1
# python fed_nlg.py --model_name llama2-7b --peft_name lora --data_name e2e_nlg --percent 0.6 --add_noise 1
# python fed_nlg.py --model_name llama2-7b --peft_name lora --data_name e2e_nlg --percent 0.7 --add_noise 1
# python fed_nlg.py --model_name llama2-7b --peft_name lora --data_name e2e_nlg --add_noise 1