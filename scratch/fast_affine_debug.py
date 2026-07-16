"""Debug affine optimization convergence."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import syntx, ants

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

# syntx affine only
rs = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
               reg_iterations=[0, 0, 0], affine_iterations=[100, 100, 0],
               similarity_metric='mattes_mi', verbose=True)

# Check parameters
m = rs['model']
print(f"\n=== Affine Parameters ===")
print(f"translation: {m.affine.translation.data}")
print(f"omega (rotation): {m.affine.omega.data}")
print(f"scale: {m.affine.scale.data}")
print(f"aniso scale: {m.affine.anisotropic_scale.data}")
print(f"shear: {m.affine.shear.data}")
print(f"T_grid:\n{m.affine.get_matrix().detach().numpy()}")

# Check affine loss trajectory
losses = [l.item() for l in m.affine_losses]
print(f"\nAffine losses (first 5): {[f'{l:.4f}' for l in losses[:5]]}")
print(f"Affine losses (last 5): {[f'{l:.4f}' for l in losses[-5:]]}")
print(f"Total affine iters: {len(losses)}")
