## This project is the numerical simulations for the paper *Variance Reduced Distributed Adaptive Stochastic Gradient Tracking for Nonconvex Optimization over Directed Networks*.

### Preparation
First, you need to clone the repository to your local machine and turn to the project directory by running the following command:

```bash
git clone https://github.com/Monee101/VRDAGT.git
cd VRDAGT
```

If you find that the datasets are not available, you can try to solve this problem by the following ways:
1. Since the datasets are larger than 100MB, you may need to use git lfs to clone the large files. If you haven't installed git lfs, you can run the following command to install:

```bash
git lfs install
```

2. Then, you can run the following command to clone the large files:
```bash
git lfs pull
```

To run the code, you need to install `python 3.9` or `higher` and the required packages listed in `requirements.txt`. You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

### A thorough explanation of the meaning and purpose of each directory is as follows:
- algorithms: Contains the implementation of the algorithms, such as VRDAGA, DAGT, GT-VR and AB-SAGA.
- data: Contains the datasets used in the experiments, all dataset have been preprocessed.
- draw-paper: Contains the visualization results of the experiments.
- draw_code: Contains the code for generating the visualization results.
- models: Contains the model definitions used in the experiments, such as neural networks and logistic regression and so on.
- utils: Contains the utility functions used in the experiments, such as communication matrix generation.
- ImageDeblu.py: **Contains the code for image deblurring tasks.** you can simply run the code to deblur the image. 
```bash
python ImageDeblu.py
```
- ML.py: **Contains the code for machine learning tasks** e.g., logistic regression, neural networks. you need input some parameters to run the code, such as the number of nodes, the number of iterations. Here is an example command to run the code:
```bash
python ML.py --nodes 3 --T 1000 --batchsize 32 --onehot --alpha 0.01
```
The parameters defind in the utils/tools.py file, you can modify them according to your needs.

To reproduce the results in the paper, you can find the parameters uesd in the experiments in each draw/xxx/log.txt file, where `xxx` is the name of the experiment, such as a9a_log (A9A logistic regression). You can run the code with the same parameters to reproduce the results.

For example, to reproduce the results of the A9A logistic regression experiment, you can run the following command:
```bash
python ML.py --nodes 10 --T 6000 --alpha 0.01 --dataset A9A --model log 
```

For logistic regression, we don't use one-hot encoding, so you can remove the `--onehot` parameter and for neural networks, you need to use the `--onehot` parameter to use one-hot encoding.

---
**If you have any questions or suggestions, please feel free to contact me. You can also open an issue on GitHub to discuss the code or report bugs.**



