# HP
Code and model of Meta-Hallucinating Prototype for Few-Shot Learning Promotion

# Requirements
- numpy  1.21.2
- scipy  1.3.0
- torch  1.6.0
- torchvision  0.7.0
- Python 3.7.3

# Train
```
- 5-way 1-shot:
python3 main.py --gpu 0 --N_shot 1 --N_reference 56 --N_reference_per_class 2 --N_generate 64 --epochs 100 --diversity_parmater 1.0 --kl_parmater 1.0 --checkpoint 5way_1shot

- 5-way 5-shot:
python3 main.py --gpu 0 --N_shot 5 --N_reference 56 --N_reference_per_class 2 --N_generate 64 --epochs 100 --diversity_parmater 1.0 --kl_parmater 1.0 --checkpoint 5way_5shot

```
## Test
```
- 5-way 1-shot:
python3 main.py --gpu 0 --N_shot 1  --evaluate 1 --resume ./5way_1shot/checkpoint.pth.tar --N_reference 56  --N_reference_per_class 2  --N_generate 64

- 5-way 5-shot:
python3 main.py --gpu 0 --N_shot 5  --evaluate 1 --resume ./5way_5shot/checkpoint.pth.tar --N_reference 56  --N_reference_per_class 2  --N_generate 64
```
