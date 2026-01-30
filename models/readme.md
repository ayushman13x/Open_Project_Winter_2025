# Model Checkpoints
This directory contains serialized `.pkl` files for Track 2 quantum tomography models.

### How to Load
To restore a model, ensure you have the `QuantumModel` class and `load_pickle` helper defined, then run:
```python
model = QuantumModel.load("models/model_track2_3.pkl")