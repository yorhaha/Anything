# Report

## Solution

We use mpi4py to create 1 master worker and N slave worker. The master worker is responsible for preparing information before task segmentation (such as the total size of the json file and the split quantity), and then assign different tasks to each slave worker. Each slave will get a unique file starting position. The slave worker will read the json file from the given position by lines and extract useful statistical information. After finishing, slave workers will submit all computation results to the master worker. The master worker use these results to calculate the table information.

## Run this code

Install conda:

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
```

Install `mpi4py` and `tabulate`:

```bash
conda install -c conda-forge mpi4py tabulate
mpirun --version
```

Run the following command in bash:

```python
mpiexec -n 4 python main.py
```
