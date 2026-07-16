"""Debug inverse warp export."""
import sys, os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, 'src')
import syntx, ants, numpy as np

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

rs = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
               reg_iterations=[10,5,0], affine_iterations=[100,100,0],
               similarity_metric='mattes_mi', verbose=False, grad_step=0.25)

print("fwdtransforms:", rs['fwdtransforms'])
print("invtransforms:", rs['invtransforms'])
print("whichtoinvert_inv:", rs.get('whichtoinvert_inv'))
print()

# Check inv warp file
for f in rs['invtransforms']:
    if f.endswith('.nii.gz'):
        w = ants.image_read(f)
        print(f"Inv warp file: {f}")
        print(f"  shape: {w.numpy().shape}")
        print(f"  origin: {w.origin}")
        print(f"  spacing: {w.spacing}")
        print(f"  max disp: {np.abs(w.numpy()).max():.3f}")
    elif f.endswith('.mat'):
        tx = ants.read_transform(f)
        print(f"Inv affine: {f}")
        print(f"  params: {tx.parameters}")

# Check: fi and mi have same geometry in this case
print(f"\nFixed origin: {fi.origin}, spacing: {fi.spacing}")
print(f"Moving origin: {mi.origin}, spacing: {mi.spacing}")

# Try applying inv_file with fixed origin instead
for f in rs['invtransforms']:
    if f.endswith('.nii.gz'):
        w = ants.image_read(f)
        w2 = ants.from_numpy(w.numpy(), origin=fi.origin, spacing=fi.spacing,
                              direction=fi.direction, has_components=True)
        import tempfile
        f2 = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False).name
        ants.image_write(w2, f2)
        
        inv_transforms2 = [t if not t.endswith('.nii.gz') else f2 for t in rs['invtransforms']]
        wi2 = ants.apply_transforms(mi, fi, inv_transforms2)
        print(f"\nInverse with fixed origin: MI={ants.image_mutual_information(mi, wi2):.4f}")

# Original inverse
wi = ants.apply_transforms(mi, fi, rs['invtransforms'])
print(f"Original inverse: MI={ants.image_mutual_information(mi, wi):.4f}")

# SyN-only test (no affine)
rs2 = syntx.syn(fi, mi, 'SyNTo', backend='pytorch',
                reg_iterations=[10,0,0], affine_iterations=[0,0,0],
                similarity_metric='mattes_mi', verbose=False, grad_step=0.25)
wi2 = ants.apply_transforms(mi, fi, rs2['invtransforms'])
print(f"\nSyN-only inverse: MI={ants.image_mutual_information(mi, wi2):.4f}")
wf2 = ants.apply_transforms(fi, mi, rs2['fwdtransforms'])
print(f"SyN-only forward: MI={ants.image_mutual_information(fi, wf2):.4f}")
