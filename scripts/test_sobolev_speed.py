import time
import ants
import syntx

fi = ants.image_read(ants.get_ants_data('r16'))
mi = ants.image_read(ants.get_ants_data('r64'))

print("Running TVF sobolev fast_smooth=False...")
t0 = time.time()
reg1 = syntx.tvf(
    fixed=fi,
    moving=mi,
    type_of_transform='SyNTVF',
    regularizer='sobolev',
    fast_smooth=False,
    optimizer='lars',
    lr=0.60,
    flow_sigma=0.5,
    total_sigma=0.05,
    cfl_momentum=0.95,
    n_time_steps=3,
    use_analytical_gradients=True,
    antisymmetric=False,
    constant_speed=True,
    constant_speed_relaxation=0.1,
    cfl_max=None,
    solver='euler',
    integration_steps_per_interval=3,
    multipoint_loss=[0.0, 0.5, 1.0],
    reg_iterations=[100, 100, 20],
    tol=1e-9,
    verbose=False
)
t1 = time.time()
print(f"  TVF sobolev fast_smooth=False completed in {t1 - t0:.2f} seconds")

print("\nRunning TVF sobolev fast_smooth=True...")
t0 = time.time()
reg2 = syntx.tvf(
    fixed=fi,
    moving=mi,
    type_of_transform='SyNTVF',
    regularizer='sobolev',
    fast_smooth=True,
    optimizer='lars',
    lr=0.60,
    flow_sigma=0.5,
    total_sigma=0.05,
    cfl_momentum=0.95,
    n_time_steps=3,
    use_analytical_gradients=True,
    antisymmetric=False,
    constant_speed=True,
    constant_speed_relaxation=0.1,
    cfl_max=None,
    solver='euler',
    integration_steps_per_interval=3,
    multipoint_loss=[0.0, 0.5, 1.0],
    reg_iterations=[100, 100, 20],
    tol=1e-9,
    verbose=False
)
t2 = time.time()
print(f"  TVF sobolev fast_smooth=True completed in {t2 - t0:.2f} seconds")
