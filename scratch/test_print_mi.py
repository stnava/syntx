import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants
import numpy as np

mi = ants.image_read(ants.get_data('r64'))
mi_np = mi.numpy()

# Print original moving row 128 from 100 to 160
print("Original moving row 128:")
print(mi_np[128, 100:160])
