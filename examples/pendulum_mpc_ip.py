import jax.numpy as jnp
import jax.random
from jax import config
from cparcon.optimal_control_problem import OCP
from cparcon.par_ip_newton import par_interior_point_optimal_control
from cparcon.utils import wrap_angle, euler
import matplotlib.pyplot as plt
import time
import timeit

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

# select platform
config.update("jax_platform_name", "cpu")
# config.update("jax_platform_name", "cuda")


def constraints(state, control):
    # control_ub = jnp.finfo(jnp.float64).max
    # control_lb = jnp.finfo(jnp.float64).min
    control_ub = 5.0
    control_lb = -5.0

    c0 = control - control_ub
    c1 = -control + control_lb
    return jnp.hstack((c0, c1))


def final_cost(state):
    goal_state = jnp.array((jnp.pi, 0.0))
    final_state_cost = jnp.diag(jnp.array([1e0, 1e-1]))
    angle, ang_vel = state
    _wrapped = jnp.hstack((wrap_angle(angle), ang_vel)) - goal_state
    err = _wrapped
    c = 0.5 * err.T @ final_state_cost @ err
    return c


def transient_cost(state, action, bp):
    goal_state = jnp.array((jnp.pi, 0.0))
    state_cost = jnp.diag(jnp.array([1e0, 1e-1]))
    action_cost = jnp.diag(jnp.array([1e-3]))
    angle, ang_vel = state
    _wrapped = jnp.hstack((wrap_angle(angle), ang_vel)) - goal_state
    err = _wrapped
    c = 0.5 * err.T @ state_cost @ err
    c += 0.5 * action.T @ action_cost @ action
    log_barrier = jnp.sum(jnp.log(-constraints(state, action)))
    return c - bp * log_barrier


def total_cost(states: jnp.ndarray, controls: jnp.ndarray, bp: float):
    ct = jax.vmap(transient_cost, in_axes=(0, 0, None))(states[:-1], controls, bp)
    cT = final_cost(states[-1])
    return cT + jnp.sum(ct)


def pendulum(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    gravity = 9.81
    length = 1.0
    mass = 1.0
    damping = 1e-3

    position, velocity = state
    return jnp.hstack(
        (
            velocity,
            -gravity / length * jnp.sin(position)
            + (action - damping * velocity) / (mass * length**2),
        )
    )


Ts = 0.01
N = 60
x0 = jnp.array([0.01, -0.01])
key = jax.random.PRNGKey(1)
u = 0.1 * jax.random.normal(key, shape=(N, 1))
dynamics = euler(pendulum, Ts)
nonlinear_problem = OCP(dynamics, constraints, transient_cost, final_cost, total_cost)

annon_par = lambda init_u, init_x0: par_interior_point_optimal_control(
    nonlinear_problem, init_u, init_x0
)

_jitted_par = jax.jit(annon_par)


def mpc_body(carry, inp):
    prev_control, prev_x = carry
    u_par, _ = _jitted_par(prev_control, prev_x)
    x_next = dynamics(prev_x, u_par[0])
    return (u_par, x_next), (x_next, u_par[0])


_, (mpc_states, mpc_controls) = jax.lax.scan(mpc_body, (u, x0), xs=None, length=400)
mpc_states = jnp.vstack((x0, mpc_states))
plt.plot(mpc_controls[:, 0])
plt.plot(mpc_states[:, 0])
plt.show()


