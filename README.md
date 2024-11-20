<h2 align="center">
    <a href="https://arxiv.org/abs/2410.10179">Is Parameter Collision Hindering Continual Learning in LLMs?</a>
</h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest updates. </h5>

<h5 align="center">
    
[![arXiv](https://img.shields.io/badge/Arxiv-Paper-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.10179) <br>
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/PKU-YuanGroup/N-LoRA/blob/main/LICENSE) 
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FPKU-YuanGroup%2FN-LoRA&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visitor&edge_flat=false)](https://hits.seeyoufarm.com)
[![GitHub issues](https://img.shields.io/github/issues/PKU-YuanGroup/N-LoRA?color=critical&label=Issues)](https://github.com/PKU-YuanGroup/N-LoRA/issues?q=is%3Aopen+is%3Aissue)

</h5>


<h3 align="center">Model Architecture</h3>
<p align="center">
    <img src="images/fig1.png" alt="(a) Orthogonal but Parameter Collision: Tasks $\tau_1$, $\tau_2$, and $\tau_3$ are mutually orthogonal but interaction within each space, resulting in parameter collision. (b) Non-collision and Orthogonal: Tasks $\tau_1$, $\tau_2$, and $\tau_3$ update only along distinct, non-conflicting subspaces, preserving prior task knowledge. (c) Performance Comparison: N-LoRA (red) and O-LoRA (blue) are compared across various metrics, with N-LoRA achieving lower collision rates, improved orthogonality, and superior average accuracy." width="600">
</p>

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

