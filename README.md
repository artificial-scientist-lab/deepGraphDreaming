🎶 https://youtu.be/Y3ywicffOj4 😴

### structure

* `neuralnet.py` defines network, training, dreaming, neuron-selector
* `train.py` trains initial NN
* `dream.py` starts dreaming for a given model
* `datagen.py` functions for generating data
* `AnalyzeDream.ipynb` to load and analyze dreaming data by hand

### setting up on the cluster

```
module load anaconda/3/2021.11
virtualenv --python=python3.9 venv
source venv/bin/activate
pip install -r requirements.txt
```

**for data generation**, configure `configs/datagen.yaml`, and run
```
sbatch startup/datagen
```
or
```
sbatch startup/datagen_parallel
```

**for training**, configure `configs/train.yaml`, and run
```
sbatch startup/train
```
models and config files are saved

**for dreaming**, configure `configs/dream.yaml`, and run
```
sbatch startup/dream
```


#### random

useful commands:
* `du -h filename` (size of file)
