import ants
import numpy as np

fixed = ants.image_read(ants.get_ants_data('r16'))
print("Fixed shape:", fixed.shape)
print("Fixed spacing:", fixed.spacing)
print("Fixed origin:", fixed.origin)
print("Fixed direction:\n", fixed.direction)

tx = ants.new_ants_transform(precision='float', dimension=2)
tx.set_parameters([1.0, 0.0, 0.0, 1.0, 5.0, -3.0])
print("tx parameters:", tx.parameters)
print("tx fixed parameters:", tx.fixed_parameters)
