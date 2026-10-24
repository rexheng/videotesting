# Weights

Perception loads checkpoints from this directory and does not download them
while it is running.

```bash
python3 -m pip install -r requirements-models.txt
python3 -m streetseg.download
python3 -m streetseg.download --signs --depth
```

That fetches YOLO26n-sem, YOLO26n, and, with the flags, the crossing-sign
checkpoint and Depth Anything V2 Small. The files stay on this machine.
