```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install matplotlib scipy sympy torchdata torchtext pandas GPUtil PyYAML pytheusQ
```

```
module load anaconda/3/2021.11
virtualenv --python=python3.9 venv
source venv/bin/activate
pip install -r requirements.txt
```

* `neuralnet.py` defines network, training, dreaming, neuron-selector
* `train.py` trains initial NN
* `dream.py` starts dreaming for a given model
* `datagen.py` functions for generating dat

useful commands:
* `du -h filename` (size of file)