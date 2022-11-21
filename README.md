```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install matplotlib scipy sympy torchdata torchtext pandas GPUtil PyYAML pytheusQ
```

* `neuralnet.py` defines network, training, dreaming, neuron-selector
* `train.py` trains initial NN
* `dream.py` starts dreaming for a given model
* `datagen.py` functions for generating dat