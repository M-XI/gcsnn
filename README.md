### README

# Installation
The only substantive requirements are PyTorch, PyTorch Vision, PyTorch Lightning and Numpy. One can install a requirements list from `requirements.txt` as below:
> pip install -r requirements.txt

# Layout
From here, `models` contains a set of models, `pipelines` contains a set of processing pipelines (for now only a basic training/evaluation routine), `utils` contains a number of useful tools (kernels, convolutions, etc.), and `ecsnn.ipynb` is the primary runner script. A folder called `data` on the same directory level will store downloaded datasets (in our case MNIST, KMNIST, FashionMNIST). A folder called `artifacts` will contain trained models as saved by PyTorch Lightning. Depending on a level of permissions/access, these may need to be created manually.

# Replication
Running the `ecsnn.ipynb` notebook will replicate all associated experiments.