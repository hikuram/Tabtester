import numpy as np
from tabicl import TabICLClassifier, TabICLRegressor

rng = np.random.default_rng(42)
X = rng.normal(size=(320, 8)).astype(np.float32)
y_reg = (0.8 * X[:, 0] - 0.3 * X[:, 1] + rng.normal(scale=0.1, size=320)).astype(np.float32)
y_clf = (X[:, 0] + X[:, 1] > 0).astype(np.int64)

for name, model, y in (
    ("regressor", TabICLRegressor(device="cpu", n_estimators=1), y_reg),
    ("classifier", TabICLClassifier(device="cpu", n_estimators=1), y_clf),
):
    print(f"Caching {name} checkpoint...")
    model.fit(X, y)

print("Checkpoint warmup complete.")
