### 1. Set up environment:

```bash
conda env create -f environment.yaml
conda activate que
```

### 2. Generate dataset:

```bash
python data.py
```

### 3. Training & Inference:

```bash
python model.py
```

**For users**: *First epoch is slow due to data processing & caching. Training from second epoch is much faster*

**For future me**: *This is not a production code*

