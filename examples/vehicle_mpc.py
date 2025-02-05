import jax.numpy as jnp
import jax.random
from jax import config
from cparcon.utils import wrap_angle, rollout, euler
from cparcon.par_admm_newton import par_admm
import matplotlib.pyplot as plt

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

# select platform
config.update("jax_platform_name", "cpu")
# config.update("jax_platform_name", "cuda")


def projection(z: jnp.ndarray) -> jnp.ndarray:
    px, py, v, phi, acc, steer = z
    car_position_consensus = jnp.hstack((px, py))

    def ellipse_projection(point_x, point_y, ellipse_a, ellipse_b, cnter_x, center_y):
        theta = jnp.arctan2(point_y - center_y, point_x - cnter_x)
        k = (ellipse_a * ellipse_b) / jnp.sqrt(
            ellipse_b**2 * jnp.cos(theta) ** 2 + ellipse_a**2 * jnp.sin(theta) ** 2
        )
        xbar = k * jnp.cos(theta) + cnter_x
        ybar = k * jnp.sin(theta) + center_y
        return jnp.hstack((xbar, ybar))

    # ellipse obstacle parameters
    ea = 5.0
    eb = 2.5
    cx = 15.0
    cy = -1.0

    # ellipse constraint
    S = jnp.diag(jnp.array([1.0 / ea**2, 1.0 / eb**2]))
    dxy = jnp.array([px - cx, py - cy])
    violation = 1 - dxy.T @ S @ dxy > 0
    projected_car_position = jnp.where(
        violation, ellipse_projection(px, py, ea, eb, cx, cy), car_position_consensus
    )

    acceleration_ub = 1.5
    acceleration_lb = -3
    steering_ub = 0.6
    steering_lb = -0.6

    projected_acc = jnp.clip(acc, acceleration_lb, acceleration_ub)
    projected_steer = jnp.clip(steer, steering_lb, steering_ub)

    projection = jnp.hstack(
        (projected_car_position, v, phi, projected_acc, projected_steer)
    )
    return projection


def stage_cost(state: jnp.ndarray, control: jnp.ndarray) -> float:
    ref = jnp.array([0.0, 0.0, 8.0, 0.0])
    state_penalty = jnp.diag(jnp.array([0.0, 1.0, 5.0, 0.0]))
    control_penalty = jnp.diag(jnp.array([1.0, 10.0]))
    c = (state - ref).T @ state_penalty @ (state - ref)
    c = c + control.T @ control_penalty @ control
    return c * 0.5


def final_cost(state: jnp.ndarray) -> float:
    ref = jnp.array([0.0, 0.0, 8.0, 0.0])
    state_penalty = jnp.diag(jnp.array([0.0, 1.0, 5.0, 0.0]))
    c = (state - ref).T @ state_penalty @ (state - ref)
    return c * 0.5


def vehicle(state: jnp.ndarray, control: jnp.ndarray):
    lf = 1.06
    lr = 1.85
    x, y, v, phi = state
    acceleration, steering = control
    beta = jnp.arctan(jnp.tan(steering * (lr / (lf + lr))))
    return jnp.hstack(
        (
            v * jnp.cos(phi + beta),
            v * jnp.sin(phi + beta),
            acceleration,
            v / lr * jnp.sin(beta),
        )
    )


simulation_setp = 0.1
dynamics = euler(vehicle, simulation_setp)
horizon = 10
penalty_parameter = 5.0
mean = jnp.array([0.0, 0.0])
sigma = jnp.array([0.01, 0.01])
x0 = jnp.array([0.0, 0.0, 5.0, 0.0])
key = jax.random.PRNGKey(1)
u = mean + sigma * jax.random.normal(key, shape=(horizon, 2))
x = rollout(dynamics, u, x0)
z = jnp.zeros((horizon, u.shape[1] + x.shape[1]))
l = jnp.zeros((horizon, u.shape[1] + x.shape[1]))

annon_par = lambda x_, u_, z_, l_: par_admm(
    stage_cost,
    final_cost,
    dynamics,
    projection,
    x_,
    u_,
    z_,
    l_,
    penalty_parameter,
)

_jitted_par = jax.jit(annon_par)


def mpc_body(carry, inp):
    prev_state, prev_controls, prev_consensus, prev_duals = carry
    states = rollout(dynamics, prev_controls, prev_state)
    _, u_admm, z_admm, l_admm, iterations = _jitted_par(
        states, prev_controls, prev_consensus, prev_duals
    )
    next_state = dynamics(prev_state, u_admm[0])
    return (next_state, u_admm, z_admm, l_admm), (next_state, u_admm[0], iterations)


_, mpc_out = jax.lax.scan(mpc_body, (x0, u, z, l), None, length=60)
mpc_states, mpc_controls, iterations = mpc_out
u = 15
v = -1.0
a = 5.0
b = 2.5
t = jnp.linspace(0, 2 * jnp.pi, 100)
plt.plot(u + a * jnp.cos(t), v + b * jnp.sin(t))
plt.plot(mpc_states[:, 0], mpc_states[:, 1])
plt.ylim(-10, 10)
plt.show()
plt.plot(mpc_states[:, 2])
plt.show()
