#!/bin/bash

# 模型名列表
# "gpt2-medium" "gpt2-large" "llama2-7b"
model_names=("gpt2-medium" "gpt2-large" "llama2-7b")
#model_names=("gpt2-medium")
#model_names=("gpt2-large")
#model_names=("llama2-7b")

# "lora" "IA3"
#peft_names=("lora" "IA3" "bottleneck")
#peft_names=("bottleneck")
peft_names=("lora")

data_names=("e2e_nlg")

# add_noise选项
#add_noises=("False" "True")
add_noises=("False")

# percent选项
percents=("None" "0.5" "0.6" "0.7")
#percents=("0.5" "0.6" "0.7")


# 遍历模型名列表
for model_name in "${model_names[@]}"; do
    # 遍历peft名列表
    for peft_name in "${peft_names[@]}"; do
        # 遍历数据名列表
        for data_name in "${data_names[@]}"; do
            # 遍历add_noise选项
            for add_noise in "${add_noises[@]}"; do
            
                # 遍历percent选项
                for percent in "${percents[@]}"; do
                    # 如果add_noise为True且percent不为None，则跳过
                    if [ "$add_noise" = "True" ] && [ "$percent" != "None" ]; then
                        continue
                    fi
                    
                    # 构建命令
                    cmd="python run_nlg.py --model_name $model_name --peft_name $peft_name --data_name $data_name"
                    if [ "$add_noise" = "True" ]; then
                        cmd="$cmd --add_noise"
                    fi
                    if [ "$percent" != "None" ]; then
                        cmd="$cmd --percent $percent"
                    fi
                    # 运行命令
                    echo
                    echo $cmd
                    eval $cmd
                    echo
                done
            done
        done
    done
done
