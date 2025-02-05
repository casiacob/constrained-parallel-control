import jax.numpy as jnp
import jax.random
from jax import config
from cparcon.utils import wrap_angle, rollout, euler
from cparcon.par_admm_newton import par_admm
from cparcon.seq_admm_newton import seq_admm
import time
import pandas as pd
import matplotlib.pyplot as plt

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

config.update("jax_platform_name", "cpu")


def projection(z):
    x_min = jnp.finfo(jnp.float64).min
    x_max = jnp.finfo(jnp.float64).max
    ub = jnp.array([x_max, x_max, x_max, x_max, 60.0])
    lb = jnp.array([x_min, x_min, x_min, x_min, -60.0])
    return jnp.clip(z, lb, ub)


def final_cost(state: jnp.ndarray) -> float:
    goal_state = jnp.array([0.0, jnp.pi, 0.0, 0.0])
    final_state_cost = jnp.diag(jnp.array([1e1, 1e1, 1e-1, 1e-1]))

    _wrapped = jnp.hstack((state[0], wrap_angle(state[1]), state[2], state[3]))
    c = 0.5 * (_wrapped - goal_state).T @ final_state_cost @ (_wrapped - goal_state)
    return c


def transient_cost(state: jnp.ndarray, action: jnp.ndarray) -> float:
    goal_state = jnp.array([0.0, jnp.pi, 0.0, 0.0])
    state_cost = jnp.diag(jnp.array([1e1, 1e1, 1e-1, 1e-1]))
    action_cost = jnp.diag(jnp.array([1e-4]))

    _wrapped = jnp.hstack((state[0], wrap_angle(state[1]), state[2], state[3]))
    c = 0.5 * (_wrapped - goal_state).T @ state_cost @ (_wrapped - goal_state)
    c += 0.5 * action.T @ action_cost @ action
    return c


def total_cost(states: jnp.ndarray, controls: jnp.ndarray):
    ct = jax.vmap(transient_cost, in_axes=(0, 0, None))(states[:-1], controls)
    cT = final_cost(states[-1])
    return cT + jnp.sum(ct)


def cartpole(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:

    # https://underactuated.mit.edu/acrobot.html#cart_pole

    gravity = 9.81
    pole_length = 0.5
    cart_mass = 10.0
    pole_mass = 1.0
    total_mass = cart_mass + pole_mass

    cart_position, pole_position, cart_velocity, pole_velocity = state

    sth = jnp.sin(pole_position)
    cth = jnp.cos(pole_position)

    cart_acceleration = (
        action + pole_mass * sth * (pole_length * pole_velocity**2 + gravity * cth)
    ) / (cart_mass + pole_mass * sth**2)

    pole_acceleration = (
        -action * cth
        - pole_mass * pole_length * pole_velocity**2 * cth * sth
        - total_mass * gravity * sth
    ) / (pole_length * cart_mass + pole_length * pole_mass * sth**2)

    return jnp.hstack(
        (cart_velocity, pole_velocity, cart_acceleration, pole_acceleration)
    )


Ts = 0.01
N = 60
dynamics = euler(cartpole, Ts)
x0 = jnp.array([0.01, wrap_angle(-0.01), 0.01, -0.01])
key = jax.random.PRNGKey(10)
u = 0.1 * jax.random.normal(key, shape=(N, 1))
x = rollout(dynamics, u, x0)
z = jnp.zeros((N, u.shape[1] + x.shape[1]))
l = jnp.zeros((N, u.shape[1] + x.shape[1]))
penalty_parameter = 0.001

annon_par = lambda x_, u_, z_, l_: par_admm(
    transient_cost,
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


_, mpc_out = jax.lax.scan(mpc_body, (x0, u, z, l), None, length=200)
mpc_states, mpc_controls, iterations = mpc_out
states = jnp.vstack((x0, mpc_states))
plt.plot(mpc_controls[:, 0] / 20)
plt.plot(states[:, 0])
plt.plot(states[:, 1])
plt.show()

