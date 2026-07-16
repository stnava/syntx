import ants
import scipy.io as sio
import tempfile
import os

tx = ants.new_ants_transform(precision='float', dimension=2)
tx.set_parameters([1.0, 0.0, 0.0, 1.0, 5.0, -3.0])

fd, tx_path = tempfile.mkstemp(suffix='.mat')
os.close(fd)
try:
    ants.write_transform(tx, tx_path)
    mat = sio.loadmat(tx_path)
    print("Keys in mat file:", mat.keys())
    for k in mat.keys():
        if not k.startswith('__'):
            print(f"{k}:")
            print(mat[k])
finally:
    if os.path.exists(tx_path):
        os.remove(tx_path)
