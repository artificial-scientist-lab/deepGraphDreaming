🎶 https://youtu.be/Y3ywicffOj4 😴

### structure

* `neuralnet.py` defines network, training, dreaming, neuron-selector
* `train.py` trains initial NN
* `dream.py` starts dreaming for a given model
* `datagen.py` functions for generating data
* `predict.ipynb` to load and analyze models by hand

### setting up on the cluster

```
module load anaconda/3/2021.11
virtualenv --python=python3.9 venv
source venv/bin/activate
pip install -r requirements.txt
```

to run data generation
```
sbatch startup_datagen
```

to run training
```
sbatch startup_train
```

#### random

useful commands:
* `du -h filename` (size of file)
