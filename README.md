# Kernel Learning in Ridge Regression Algorithm with Warm Start

## Paper:
[Kernel Learning in Ridge Regression “Automatically” Yields Exact Low Rank Solution](https://arxiv.org/abs/2310.11736).

## Requirement:
- python 3.9
- pytorch==1.13.0
- scipy==1.10.1
- tqdm==4.64.1
- pandas==2.1.0
- matplotlib==3.8.0
- scikit-learn==1.2.2

## Example code:
1. for simulation
```bash
python3 exprep0.py --s 1 --l 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1
                   --n 300 --d 50 --args 1
python3 exprep0.py --s 1 --l 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.2,2.4,2.6,2.8,3
                   --n 300 --d 50 --args 3
python3 exprep0.py --s 1 --l 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.2,2.4,2.6,2.8,3
                   --n 300 --d 50 --args 4 --alpha 0.0001
```
2. for existing data
```bash
# --data train,test
python3 exprep0.py --l 1,0.5 --data ./example simulated data/example_train.csv,./example simulated data/example_test.csv
# --data train,test,validate
python3 exprep0.py --l 1,0.5 --data ./example simulated data/example_train.csv,./example simulated data/example_test.csv,./example simulated data/example_test.csv
```

Arguments:
1. for data
- for existing data
   - `--data`: data path (built in: **csv** data paths with **header** and with **last column as y**, using comma to separate train and test path, i.e. train,test/train,test,validate). Default: None. \
      Notice: When this value is assigned, all the rest arguments for simulation data will be ignored!

- for simulation data
   - `--s`: random seed of simulation repetition (the experiments in our paper are repeated 100 times by applying `$SLURM_ARRAY_TASK_ID` to assign 1-100 of this argument). Default: `--s 1`.
   - `--n`: sample size. Default: `--n 300`.
   - `--d`: sample dimension. Default: `--d 50`.
   - `--e`: noise size. Default: `--e 0.1`.
   - `--args`: function index in paper (built in: 1/2/3/4/5, see [section 7.1 function f](https://arxiv.org/pdf/2310.11736.pdf)). Default: `--args 3`.
   - `--xdis`: distribution of X index in paper (built in: 1/2/3, which indicates Gau/Unif/Ber respectively). Default: `--xdis 1`.
   - `--rho`: distribution of X parameter: correlation for `--xdis 1` and probability equals to one for `--xdis 3`. Default: `--rho 0.`

2. for model
- `--l`: lambda sequence using comma to separate (this argument affects warm start, recommend to use dense grids). E.g. `--l 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1`.
- `--ker`: kernel function (built in: Gau/Gaudiag/linear/cubic). Default: `--ker Gau`.

3. for optimization
- `--iter`: maximum iteration number. Default: `--iter 2000`.
- `--lr`: initial learning rate. Default: `--lr 0.1`.
- `--alpha`: parameter for Armijo rule. Default: `--alpha 0.001`.
- `--beta`: parameter for Armijo rule. Default: `--beta 0.5`.
- `--tol`: parameter for tolerance. Default: `--tol 0.001`.

4. for classification
- `--cl`: whether a binary classification problem (built in: 0/1, using y labels 0,1). Default: `--cl 0`.

## Attribution:
Certain portions of this codebase are referred to the work of [recursive feature machines](https://github.com/aradha/recursive_feature_machines).
