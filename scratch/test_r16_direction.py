import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import ants

fi = ants.image_read(ants.get_data('r16'))
mi = ants.image_read(ants.get_data('r64'))

print("r16 origin:", fi.origin)
print("r16 spacing:", fi.spacing)
print("r16 direction:\n", fi.direction)

print("\nr64 origin:", mi.origin)
print("r64 spacing:", mi.spacing)
print("r64 direction:\n", mi.direction)
