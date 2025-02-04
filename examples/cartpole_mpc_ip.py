import jax.numpy as jnp
import jax.random
from jax import config
from cparcon.optimal_control_problem import OCP
from cparcon.par_ip_newton import par_interior_point_optimal_control
from cparcon.utils import wrap_angle, euler
import matplotlib.pyplot as plt

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

config.update("jax_platform_name", "cpu")


def constraints(state, control):
    control_ub = 60.0
    control_lb = -60.0

    c0 = control[0] - control_ub
    c1 = -control[0] + control_lb
    return jnp.hstack((c0, c1))


def final_cost(state: jnp.ndarray) -> float:
    goal_state = jnp.array([0.0, jnp.pi, 0.0, 0.0])
    final_state_cost = jnp.diag(jnp.array([1e1, 1e1, 1e-1, 1e-1]))

    _wrapped = jnp.hstack((state[0], wrap_angle(state[1]), state[2], state[3]))
    c = 0.5 * (_wrapped - goal_state).T @ final_state_cost @ (_wrapped - goal_state)
    return c


def transient_cost(state: jnp.ndarray, action: jnp.ndarray, bp) -> float:
    goal_state = jnp.array([0.0, jnp.pi, 0.0, 0.0])
    state_cost = jnp.diag(jnp.array([1e1, 1e1, 1e-1, 1e-1]))
    action_cost = jnp.diag(jnp.array([1e-4]))

    _wrapped = jnp.hstack((state[0], wrap_angle(state[1]), state[2], state[3]))
    c = 0.5 * (_wrapped - goal_state).T @ state_cost @ (_wrapped - goal_state)
    c += 0.5 * action.T @ action_cost @ action
    log_barrier = jnp.sum(jnp.log(-constraints(state, action)))
    return c - bp * log_barrier


def total_cost(states: jnp.ndarray, controls: jnp.ndarray, bp: float):
    ct = jax.vmap(transient_cost, in_axes=(0, 0, None))(states[:-1], controls, bp)
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
x0 = jnp.array([0.01, -0.01, 0.01, -0.01])
key = jax.random.PRNGKey(1)
u = 0.1 * jax.random.normal(key, shape=(N, 1))
dynamics = euler(cartpole, Ts)
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


_, (states, controls) = jax.lax.scan(mpc_body, (u, x0), xs=None, length=400)
states = jnp.vstack((x0, states))
plt.plot(controls[:, 0] / 20)
plt.plot(states[:, 0])
plt.plot(states[:, 1])
plt.show()
