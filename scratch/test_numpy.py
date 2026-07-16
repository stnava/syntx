import ants
import numpy as np

img = ants.image_read(ants.get_ants_data('r16'))
arr = img.numpy()
print(img.shape)
print(arr.shape)
