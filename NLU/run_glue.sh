#!/bin/bash

# 模型名列表
model_names=('bert-base-uncased')
# model_names=('roberta-base')

# peft名列表
peft_names=('lora' 'IA3' 'bottleneck')

# 数据名列表
data_names=("cola" "mrpc" "rte" "sst2" "stsb")
# data_names=("mnli" "qqp")

# add_noise选项
add_noises=("False" "True")

# percent选项
percents=("None" "0.5" "0.6" "0.7")


# 遍历模型名列表
for model_name in "${model_names[@]}"; do
    # 遍历peft名列表
    for peft_name in "${peft_names[@]}"; do
        # 遍历数据名列表
        for data_name in "${data_names[@]}"; do
            # 遍历percent选项
            for percent in "${percents[@]}"; do
                # 遍历add_noise选项
                for add_noise in "${add_noises[@]}"; do
                    # 构建命令
                    cmd="python run_glue.py --model_name $model_name --peft_name $peft_name --data_name $data_name"
                    if [ "$add_noise" = "True" ]; then
                        cmd="$cmd --add_noise"
                    fi
                    if [ "$percent" != "None" ]; then
                        cmd="$cmd --percent $percent"
                    fi
                    # 运行命令
                    echo
                    echo $cmd
                    # eval $cmd
                    echo
                done
            done
        done
    done
done