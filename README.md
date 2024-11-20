# Is Parameter Collision Hindering Continual Learning in LLMs?
Code for the paper "Is Parameter Collision Hindering Continual Learning in LLMs?", exploring novel approaches to address parameter collision issues in large language models for continual learning.


# Sn-LoRA

- This repo releases our implementation for the N-LoRA model.
- It is built based on the pretrained T5-large model, and finetuned on our data.

## Data Preparation

Due to the 10MB size limit for supplementary materials, the dataset used in this work cannot be directly included. However, you can refer to the O-LoRA repository (which can be found in the original O-LoRA paper) for the required data. Please follow the steps below to prepare the dataset:

1. Navigate to the O-LoRA repository (which can be found in the original O-LoRA paper) and locate the `O-LoRA-main/CL_Benchmark` folder.
2. Copy the benchmark data from the `CL_Benchmark` folder in the O-LoRA repository.
3. Place the copied data into the `N-LoRA/CL_Benchmark` folder in this repository.

By completing these steps, you will have the necessary data to reproduce our experiments.

## Setup

You can install the required libraries by running 

```
pip install -r requirements.txt
```

You are also required to download the t5-large model from huggingface, put it to the folder named ```initial_model```, and rename the model folder as 't5-large'.

LLaMA is also supported. You can put your llama model to the folder named ```initial_model``` and rename the model folder as 'llama'.


## Training and Evaluation

For t5-large:

You can reproduce our experiments of order 1 & 2 & 3 & 4 & 5 & 6 by simply running

order1:

```
bash scripts/order_1.sh> logs_and_outputs/order_1/logs/train_and_infer.log 2>&1 &
```

order2:

```
bash scripts/order_2.sh> logs_and_outputs/order_2/logs/train_and_infer.log 2>&1 &
```

order3:

```
bash scripts/order_3.sh> logs_and_outputs/order_3/logs/train_and_infer.log 2>&1 &
```

order4:

```
bash scripts/order_4.sh> logs_and_outputs/order_4/logs/train_and_infer.log 2>&1 &
```

order5:

```
bash scripts/order_5.sh> logs_and_outputs/order_5/logs/train_and_infer.log 2>&1 &
```

order6:

```
bash scripts/order_6.sh> logs_and_outputs/order_6/logs/train_and_infer.log 2>&1 &
```


The model you have trained will be saved in ```logs_and_outputs/order_1(2 or 3 or 4 or 5 or 6)/outputs```.

The result of each task will be saved in ```logs_and_outputs/order_1(2 or 3 or 4 or 5 or 6)/outputs/TASK_NAME/predict_results.json```.

You can also check the logs during training and infering in  ```logs_and_outputs/order_1(2 or 3 or 4 or 5 or 6)/logs/train_and_infer.log```

For LLaMA:

order1:

```
bash scripts_llama/order_1.sh> logs_and_outputs_llama/order_1/logs/train_and_infer.log 2>&1 &
```

order2:

```
bash scripts_llama/order_2.sh> logs_and_outputs_llama/order_2/logs/train_and_infer.log 2>&1 &
```

order3:

```
bash scripts_llama/order_3.sh> logs_and_outputs_llama/order_3/logs/train_and_infer.log 2>&1 &
```

