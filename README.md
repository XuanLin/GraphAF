# GraphAF Anonymous Source Code

This is the anonymous code for the ICLR submission:  

**GRAPHAF: A FLOW-BASED AUTOREGRESSIVE MODEL FOR MOLECULAR GRAPH GENERATION**

Note that the current code may contain bugs and is kind of messy, because we haven't reorganized the code due to the limited time.

## 1. Install Environment

* To install the rdkit, please refer to the official website. We highly recommend using anaconda3

  `conda create -c rdkit -n rdkit_env_test rdkit`

* Activate your environment: `source activate rdkit_env_test `

* Install networkx

  `conda install networkx`

* Install torch, you may need to choose the correct version depending on the configuration of your machine

  `conda install pytorch torchvision cudatoolkit=10.1 -c pytorch`

## 2. Preprocess Dataset

We provide zinc250k dataset in `./dataset/250k_rndm_zinc_drugs_clean_sorted.smi`. We also select 800 molecules with lowest penalized logP scores and provide the data in `./dataset/zinc800.smi_logp`

- To preprocess zinc250 dataset. run `preprocess_data.sh`. It will convert smiles of molecules into node/adjacency features. The preprocessed features are stored in `./data_preprocessed`. I will take 2-5 minutes roughly.

## 3.Pretraining(Density Modeling and Generation)

* Run `train_parallel.sh`. 
* Checkpoints of each epochs will be save in `./save_pretrain`. 
* We will also generate 100 molecules at the end of each epoch and save these molecules in `./save_pretrain/model_name/mols	`
* Current implementation support multiple GPU for data parallel. You can set `CUDA_VISIBLE_DEVICES` in the script for gpu-parallel training.

## 4.Generation

* Given a checkpoint, you can generate molecules using the script `generate.sh`. 
* We provide the checkpoint of a well-pretrained GraphAF in `./good_ckpt/checkpoint277`. 

* The generated molecules will be saved in `./mols`



## 5. Constrained Property Optimization

First we need to preprocess the molecules with `lowest` plogp in ZINC

```
python preprocess.py zinc250k_800 ./dataset/zinc800.smi_logp \
	./data_preprocessed/zinc_800 0
	
cp ./dataset/zinc800.smi_logp ./data_preprocessed/zinc_800.smi_logp
```



### 1. RL Finetune

* First run `constrain_optim_rl.sh` to finetune the pretrained GraphAF
* The results will be save in `reinforce_co_rl.......`. Then we can load the ckpt to optimize each molecule.

### 2. Optimize Molecule

* we provide checkpoint of the tuned GraphAF for constrained optimization in `./good_ckpt/checkpoint_co`
* To optimize the molecule, just run `constrain_optim_rl_ckpt.sh`.
  *  It consumes three args: (optim_start, optim_end, repeat_time)
  * For example, `./constrain_optim_rl_ckpt.sh 0 5 200` will optimize the first 5 molecules and each molecule will be optimized for 200 times.
* the results will be save in `co_res`



## 6. Property Optimization

The details of this task will be elaborated upon the official release of  **GraphAF**.







