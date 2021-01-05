# MTL-KGC
This is PyTorch implementation of the [Multi-Task Learning for Knowledge Graph Completion with Pre-trained Language Models](https://www.aclweb.org/anthology/2020.coling-main.153/).

**Train**

Train multitask learning with link prediction (LP), relation prediction (RP), and relevance ranking (RR).
```
python run_bert_multitask.py \
    --do_train \
    --task_list lp,rp,rr \
    --data_dir ./data/wn18rr \
    --bert_model bert-base-cased \
    --max_seq_length 128 \
    --output_dir ./output_dir \
    --num_train_epochs 5.0 \
    --learning_rate=2e-5 \
    --tb_log_dir=runs/log_dir
```
**Train link prediction only**

```
python run_bert_multitask.py \
    --do_train \
    --task_list lp \
    --data_dir ./data/wn18rr \
    --bert_model bert-base-cased \
    --max_seq_length 128 \
    --output_dir ./output_dir \
    --num_train_epochs 5.0 \
    --learning_rate=2e-5 \
    --tb_log_dir=runs/log_dir
```
**Evaulation**
```
python run_bert_multitask.py \
    --do_eval \
    --data_dir ./data/wn18rr \
    --bert_model {test_model} \
    --max_seq_length 128 \
    --output_dir ./output_dir \
```
**Performance**

WN18RR

|  | MRR | MR | HITS@1 | HITS@3 | HITS@10 |
|-------------|-------------|-------------|-------------|-------------|-------------|
| LP (Yao et al, 2019) | 0.219 | 108 | 0.095 | 0.243 | 0.497 |
| LP + RP | 0.302 | 112 | 0.177 | 0.353 | 0.560 |
| LP + RR | 0.277 | 97 | 0.130 | 0.341 | 0.576 |
| LP + RP + RR| 0.331 | 89 | 0.203 | 0.383 | 0.597 |

FB15k-237

|  | MRR | MR | HITS@1 | HITS@3 | HITS@10 |
|-------------|-------------|-------------|-------------|-------------|-------------|
| LP (Yao et al, 2019) | 0.237 | 145 | 0.144 | 0.260 | 0.427 |
| LP + RP | 0.262 | 138 | 0.169 | 0.289 | 0.447 |
| LP + RR | 0.247 | 143 | 0.154 | 0.272 | 0.434 |
| LP + RP + RR| 0.267 | 132 | 0.172 | 0.298 | 0.458 |
