import ants
fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))
reg = ants.registration(fi, mi, 'SyN', syn_iterations=[20, 0, 0])
print(reg['fwdtransforms'])
