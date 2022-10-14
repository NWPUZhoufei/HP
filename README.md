# HP
Code and model of Meta-Hallucinating Prototype for Few-Shot Learning Promotion

# Requirements
- numpy  1.21.2
- scipy  1.3.0
- torch  1.6.0
- torchvision  0.7.0
- Python 3.7.3


## Preparation
Process the base dataset `base_data_process.py`

## Train
```
python train.py --N_way 10 --N_shot 1 --N_query 19 

```
## Test
```
python test.py --pretrain_path your model  --data_name PaviaU  --test_way 9  --test_shot 1  --run_number 666 

```
