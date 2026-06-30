### 检查 npz 形状

python3 - <<'PY'
import numpy as np
path = "collected_data/你的文件名.npz"
data = np.load(path, allow_pickle=False)
for k in data.files:
    arr = data[k]
    print(k, arr.shape, arr.dtype)
PY