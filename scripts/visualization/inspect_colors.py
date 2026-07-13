from PIL import Image
import numpy as np
from collections import Counter
import sys
from echonet.paths import VISUALIZATIONS_OUTPUT_DIR

path = sys.argv[1]

img = Image.open(path).convert("RGB")
arr = np.array(img)

pixels = arr.reshape(-1, 3)
counts = Counter(map(tuple, pixels))

print("Top 25 RGB colors:")
for rgb, count in counts.most_common(25):
    print(rgb, count)
