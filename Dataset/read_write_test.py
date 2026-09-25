import numpy as np

video = np.random.randint(0, 255, (10, 64, 64, 3), dtype=np.uint8)
audio = np.random.rand(128, 100).astype(np.float32)
label = 1