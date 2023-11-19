# Kernel Learning in Ridge Regression

## Introduction:
We consider kernels parameterized by $\Sigma$
$$(x,x')$$
where $\norm{x-x'}_\Sigma = \sqrt{(x-x')^T \Sigma (x-x')}$, and $\phi$ is a real-valued function so that $k_\Sigma$ is a kernel for every positive semidefinite $\Sigma$. 
An example is the Gaussian kernel where $\phi(z) = \exp(-z)$.

We study a variant of kernel ridge regression where we also optimize the 
reproducing kernel Hilbert space $\mathcal{H}_\Sigma$ whose kernel is given by $k_\Sigma$:
$$	\mathop{\rm minimize}_{f, \gamma, \Sigma} 
		~~~\half \mathbb{E}_n \left[(Y - f(X) - \gamma)^2\right] + \frac{\lambda}{2} \norm{f}_{\mathcal{H}_\Sigma}^2
	~~~
	\mathop{\rm subject\;to}  ~~~~\Sigma \succeq 0
$$

If you are interested in more details, please refer to our paper: [Kernel Learning in Ridge Regression “Automatically” Yields Exact Low Rank Solution](https://arxiv.org/abs/2310.11736).

## Python Environment:
- python 3.9
- pytorch==1.13.0
- scipy==1.10.1
- tqdm==4.64.1
- pandas==2.1.0
- matplotlib==3.8.0
- scikit-learn==1.2.2

## Commands for reproducing figures in our paper (https://arxiv.org/abs/2310.11736):
```bash
# see arguments --s for random seeds to reproduce the results in paper and other arguments for detailed information
# Figures 1 and 2
python3 exprep0.py --l 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1
                   --n 300 --d 50 --args 1
python3 exprep0.py --l 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.2,2.4,2.6,2.8,3
                   --n 300 --d 50 --args 3
python3 exprep0.py --l 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.2,2.4,2.6,2.8,3
                   --n 300 --d 50 --args 4 --alpha 0.0001

# Figure 3
python3 exprep0.py --l 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1
                   --n 300 --d 50 --args 1 --rho 0.5
python3 exprep0.py --l 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1
                   --n 300 --d 50 --args 2 --rho 0.5 --alpha 0.0001
python3 exprep0.py --l 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.2,2.4,2.6,2.8,3
                   --n 300 --d 50 --args 3 --rho 0.5 --alpha 0.0001
python3 exprep0.py --l 0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3
                   --n 300 --d 50 --args 4 --rho 0.5 --alpha 0.0001

# Figure 4
python3 exprep0.py --l 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1 
                   --n 300 --d 50 --args 3 --xdis 2 --alpha 0.0001
python3 exprep0.py --l 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.2,2.4,2.6,2.8,3
                   --n 300 --d 50 --args 3 --xdis 3 --rho 0.5

# Figure 5
python3 exprep0.py --l 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1 
                   --n 300 --d 50 --args 5

# Figure 6
python3 exprep0.py --l 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1 
                   --n 300 --d 50 --args 1 --ker linear
```

## Commands for using kernel learning on the example data (or on your own data): 
```bash
# see Arguments --data for detailed information on using your own data
# for training and testing (--data train path,test path)
python3 exprep0.py --l 1,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05 
                   --data ./example simulated data/example_train.csv,./example simulated data/example_test.csv
# for training, validation and testing (--data train path,test path,validate path)
python3 exprep0.py --l 1,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05 
                   --data ./example simulated data/example_train.csv,./example simulated data/example_test.csv,./example simulated data/example_test.csv
```

## Arguments:
1. Objective
- for using kernel learning on example data (or on your own data)
   - `--data`: data path (built in: **csv** data paths with **header** and with **last column as y**, using comma to separate train and test path, i.e. train,test/train,test,validate). Default: None. \
      Notice: When this value is assigned, all the rest arguments for simulation data will be ignored!

- for reproducing figures in the paper
   - `--s`: random seed of simulation repetition (the experiments in our paper are repeated 100 times by applying `$SLURM_ARRAY_TASK_ID` to assign 1-100 of this argument). Default: `--s 1`.
   - `--n`: sample size. Default: `--n 300`.
   - `--d`: sample dimension. Default: `--d 50`.
   - `--e`: noise size. Default: `--e 0.1`.
   - `--args`: function index in paper (built in: 1/2/3/4/5, see Section 7.1 function f in the paper). Default: `--args 3`.
   - `--xdis`: distribution of X index in paper (built in: 1/2/3, which indicates Gau/Unif/Ber respectively). Default: `--xdis 1`.
   - `--rho`: distribution of X parameter: correlation for `--xdis 1` and probability equals to one for `--xdis 3`. Default: `--rho 0.`

2. Tuning parameters on the kernel learning model
- `--l`: lambda sequence using comma to separate (this argument affects warm start, recommend to use dense grids). E.g. `--l 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1`.
- `--ker`: kernel function (built in: Gau/Gaudiag/linear/cubic). Default: `--ker Gau`.

3. Tuning parameters in the optimization procedure
- `--iter`: maximum iteration number. Default: `--iter 2000`.
- `--lr`: initial learning rate per iteration. Default: `--lr 0.1`.
- `--alpha`: parameter for Armijo rule. Default: `--alpha 0.001`.
- `--beta`: parameter for Armijo rule. Default: `--beta 0.5`.
- `--tol`: parameter for tolerance. Default: `--tol 0.001`.

The optimization procedure is as follows. We derive an explicit formula for
$J_n(\Sigma) = \mathop{\rm minimize}_{f, \gamma} \half \E_n [(Y - f(X)- \gamma)]^2 + \frac{\lambda}{2} \norm{f}_{\mathcal{H}_\Sigma}^2$
and evaluate the gradient $\grad J_n(\Sigma)$. To then minimize $J_n(\Sigma)$ subject to $\Sigma \succeq 0$,
we perform gradient descent with projection
onto the semidefinite cone $\mathcal{C} = \{\Sigma: \Sigma \succeq 0\}$ per iteration, using the Armijo rule to search each 
iteration's stepsize. We terminate gradient descent
when the ratio between the difference of consecutive iterates, measured by the Frobenius norm, 
and the stepsize is below the tolerance $\Delta > 0$. The algorithm is always initialized at a diagonal matrix with diagonal 
entry $1/p$.


## Attribution:
Certain portions of this codebase are referred to the work of [recursive feature machines](https://github.com/aradha/recursive_feature_machines).
