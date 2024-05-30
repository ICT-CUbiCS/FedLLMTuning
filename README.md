## conda环境安装
```sh
conda env create -f FedLLMTuning.yml
```

## e2e_metrics评估代码下载
e2e_metrics用于评估NLG任务，指标有BLEU、NIST、Metor、Rouge-L、CIDEr。

下载[e2e_metrics.zip](https://drive.google.com/file/d/1F33habvGWYbpI7O4d1EaGH3NrG_bIY4_/view?usp=drive_link)文件，然后将此文件解压到主目录，得到主目录下的e2e_metrics目录。

## 数据和模型准备
推荐使用自定义的模型目录和数据目录，因为使用huggingface自定义的数据和模型路径有些乱。
```sh
# 模型准备
## 建立模型存放目录(不一定要在项目中建立，可以在其他有富余的空间中建立)
mkdir -p ./LargeModel

## 从 (https://huggingface.co/) 下载预训练基础模型（google-bert/bert-base-uncased、FacebookAI/roberta-base、openai-community/gpt2-medium、openai-community/gpt2-large、meta-llama/Llama-2-7b）
ls ./LargeModel
> bert-base-uncased roberta-base gpt2-medium gpt2-large llama2-7b

# 数据准备
## 建立数据目录
mkdir -p ./Datasets

## 建立NLU数据集
### NLU数据集
mkdir -p ./Datasets/glue
python
>>> import datasets
### 重复上述命令，分别下载并保存NLU数据（'cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli'）
>>> data = datasets.load_dataset("glue", "cola")
>>> data.save_to_disk("./Datasets/glue/cola")
...
...

### NLG数据集
cd ./NLG
mkdir -p ./e2e
### 从 https://drive.google.com/file/d/1G47rc39io-DGPjJekoAQ82_5RxWKlD05/view?usp=drive_link 下载数据文件，并将数据文件解压到./e2e目录
```

## 修改配置文件信息
进入项目主目录, 打开config.py文件
```
MODEL_DIR=os.path.join("/.../LargeModel")
DATA_DIR=os.path.join("/.../Datasets")
```

注：下面运行过程的sh脚本中，add_noises表示使用嵌入差分隐私，percents表示使用剪枝算法，根据具体需要进行调整。

## NLU代码运行
进入NLU目录
1. 检查run_glue.py, fed_glue.py
```python
... run_glue.py
if __name__ == "__main__":
    
    # main_test()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--peft_name", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--add_noise", action="store_true", default=False)
    parser.add_argument("--percent", type=float, default=None)
    
    args = parser.parse_args()
    print("\nargs:")
    print(args)
    main(args.model_name, args.peft_name, args.data_name, args.add_noise, args.percent)
    
    # gather_eval_results()
```
```python
... fed_glue.py
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="cola")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--peft_name", type=str, default="lora")
    parser.add_argument("--add_noise", type=bool, default=False)
    parser.add_argument("--percent", type=float, default=None)
    args = parser.parse_args()
    print(args)
    
    main(**vars(args))
    plot_the_eval_metric(**vars(args))
    
    # run()
    
    # gather_eval_metrics()
```
2. 运行run_glue.sh
```sh
## 设置后台运行，运行输出到日志文件中
nohup sh ./run_glue.sh > run_glue.log 2>&1 &
nohup sh ./fed_glue.sh > fed_glue.log 2>&1 &

## 观察运行时的输出信息
tail -f ./run_glue.log
tail -f ./fed_glue.log
```
3. 收集实验结果,修改run_glue.py,fed_glue.py
```python
... run_glue.py
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
```
```python
... fed_glue.py
if __name__ == "__main__":
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--data_name", type=str, default="cola")
    # parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    # parser.add_argument("--peft_name", type=str, default="lora")
    # parser.add_argument("--add_noise", type=bool, default=False)
    # parser.add_argument("--percent", type=float, default=None)
    # args = parser.parse_args()
    # print(args)
    
    # main(**vars(args))
    # plot_the_eval_metric(**vars(args))
    
    # run()
    
    gather_eval_metrics()
```
```sh
python run_glue.py

python fed_glue.py
```
4. 查看实验结果
```sh
cat ./eval_results.csv

cat ./fl_eval_metrics.csv
```

## NLG代码运行
进入NLG目录
1. 检查run_nlg.py,fed_nlg.py
```python
...run_nlg.py
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
    
    train_with_trainer(**vars(args))
        # output_dir = get_output_dir(
    #     args.model_name, args.peft_name, args.data_name, args.add_noise, args.percent
    # )
    # assert os.path.exists(output_dir), f"output_dir: {output_dir} is not exist"
    # eval_with_generate(
    #     final_model_path=os.path.join(output_dir, "client_0", "final_model"),
    #     data_name=args.data_name, peft_name=args.peft_name
    # )
    
    # gather_eval_results()
    # print_model_size(**vars(args))
```
```python
...fed_nlg.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="e2e_nlg")
    parser.add_argument("--model_name", type=str, default="gpt2-medium")
    parser.add_argument("--peft_name", type=str, default="lora")
    parser.add_argument("--add_noise", type=bool, default=False)
    parser.add_argument("--percent", type=float, default=None)
    args = parser.parse_args()
    print(args)
    
    main(**vars(args))
    
    # gather_eval_metrics()
```
2. 运行run_nlg.sh
```sh
## 设置后台运行，运行输出到日志文件中
nohup sh run_nlg.sh > run_nlg.log 2>&1 &
nohup sh fed_nlg.sh > fed_nlg.log 2>&1 &

## 观察运行时的输出信息
tail -f run_nlg.log
tail -f fed_nlg.log
```
3. 收集实验结果，修改run_nlg.py,fed_nlg.py
```python
... run_nlg.py
if __name__ == "__main__":
    
    # main_test()
    
    # get args from command line using argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name", type=str, required=True)
    # parser.add_argument("--peft_name", type=str, required=True)
    # parser.add_argument("--data_name", type=str, required=True)
    # parser.add_argument("--add_noise", action="store_true", default=False)
    # parser.add_argument("--percent", type=float, default=None)
    
    # args = parser.parse_args()
    # print("\nargs:")
    # print(args)
    
    # train_with_trainer(**vars(args))
    # output_dir = get_output_dir(
    #     args.model_name, args.peft_name, args.data_name, args.add_noise, args.percent
    # )
    # assert os.path.exists(output_dir), f"output_dir: {output_dir} is not exist"
    # eval_with_generate(
    #     final_model_path=os.path.join(output_dir, "client_0", "final_model"),
    #     data_name=args.data_name, peft_name=args.peft_name
    # )
    
    gather_eval_results()
```
```python
...fed_nlg.py
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--data_name", type=str, default="e2e_nlg")
    # parser.add_argument("--model_name", type=str, default="gpt2-medium")
    # parser.add_argument("--peft_name", type=str, default="lora")
    # parser.add_argument("--add_noise", type=bool, default=False)
    # parser.add_argument("--percent", type=float, default=None)
    # args = parser.parse_args()
    # print(args)
    
    # main(**vars(args))
    
    gather_eval_metrics()
```
4. 查看实验结果
```sh
cat ./eval_results.csv

cat ./fl_eval_metrics.csv
```
