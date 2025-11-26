# Causal Imitation Learning (Under L2.5 layer)

This repository contains the implementation of a sequential imitation learning model under L2.5 layer.

---

## Experiments 
We compare the imitation performance between:
1. The legacy π-BD policy for imitating the expert L2.5 actions (baseline).
2. The L2.5-aware strategy in the same environment (proposed method).

The experiments were conducted on the following two graphs, with 5,000 SCMs and a sample size of 1,000 for each SCM.
The evaluation metric is the mean squared error (MSE) of \( Y \).


## Usage
### Graph 1
```
python run_graph1_layer25.py
```

### Graph 2
```
python run_graph2_layer25.py
```