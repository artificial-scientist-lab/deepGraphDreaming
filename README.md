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

**for data generation**, configure `config_datagen.yaml`, and run
```
sbatch startup_datagen
```
or
```
sbatch startup_datagen
```

**for training**, configure `config_train.yaml`, and run
```
sbatch startup_train
```
models and config files are saved

**for dreaming**, configure `config_dream.yaml`, and run
```
sbatch startup_dream
```


#### random

useful commands:
* `du -h filename` (size of file)
