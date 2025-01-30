import jax.numpy as jnp
import jax.random
from jax import config
from cparcon.utils import wrap_angle, rollout, euler
from cparcon.par_admm_newton import par_admm
from cparcon.seq_admm_newton import seq_admm
import time
import pandas as pd

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

config.update("jax_platform_name", "cuda")


def projection(z):
    control_ub = jnp.array([jnp.inf, jnp.inf, 5.0])
    control_lb = jnp.array([-jnp.inf, -jnp.inf, -5.0])
    return jnp.clip(z, control_lb, control_ub)


def final_cost(state):
    goal_state = jnp.array((jnp.pi, 0.0))
    final_state_cost = jnp.diag(jnp.array([2e1, 1e-1]))
    angle, ang_vel = state
    _wrapped = jnp.hstack((wrap_angle(angle), ang_vel)) - goal_state
    err = _wrapped
    c = 0.5 * err.T @ final_state_cost @ err
    return c


def transient_cost(state, action):
    goal_state = jnp.array((jnp.pi, 0.0))
    state_cost = jnp.diag(jnp.array([2e1, 1e-1]))
    action_cost = jnp.diag(jnp.array([1e-3]))
    angle, ang_vel = state
    _wrapped = jnp.hstack((wrap_angle(angle), ang_vel)) - goal_state
    err = _wrapped
    c = 0.5 * err.T @ state_cost @ err
    c += 0.5 * action.T @ action_cost @ action
    return c


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


penalty_parameter = 1.0
Ts = [0.05, 0.025, 0.0125, 0.01, 0.005, 0.0025, 0.00125, 0.001]
N = [20, 40, 80, 100, 200, 400, 800, 1000]
par_time_means = []
par_time_medians = []
seq_time_means = []
seq_time_medians = []


for sampling_period, horizon in zip(Ts, N):
    par_time_array = []
    seq_time_array = []
    downsampling = 1
    dynamics = euler(pendulum, sampling_period)

    x0 = jnp.array([wrap_angle(0.1), -0.1])
    key = jax.random.PRNGKey(1)
    u = 0.1 * jax.random.normal(key, shape=(horizon, 1))
    x = rollout(dynamics, u, x0)
    z = jnp.zeros((horizon, u.shape[1] + x.shape[1]))
    l = jnp.zeros((horizon, u.shape[1] + x.shape[1]))

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
    annon_seq = lambda x_, u_, z_, l_: seq_admm(
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
    _jitted_seq = jax.jit(annon_seq)

    _, _, _, _, _ = _jitted_par(x, u, z, l)
    _, _, _, _, _ = _jitted_seq(x, u, z, l)
    for i in range(10):
        start = time.time()
        _, u_par_admm, _, _, _ = _jitted_par(x, u, z, l)
        jax.block_until_ready(u_par_admm)
        end = time.time()
        par_time = end - start

        start = time.time()
        _, u_seq_admm, _, _, _ = _jitted_seq(x, u, z, l)
        jax.block_until_ready(u_seq_admm)
        end = time.time()
        seq_time = end - start

        par_time_array.append(par_time)
        seq_time_array.append(seq_time)

    par_time_means.append(jnp.mean(jnp.array(par_time_array)))
    par_time_medians.append(jnp.median(jnp.array(par_time_array)))
    seq_time_means.append(jnp.mean(jnp.array(seq_time_array)))
    seq_time_medians.append(jnp.median(jnp.array(seq_time_array)))

par_time_means_arr = jnp.array(par_time_means)
par_time_medians_arr = jnp.array(par_time_medians)
seq_time_means_arr = jnp.array(seq_time_means)
seq_time_medians_arr = jnp.array(seq_time_medians)


df_mean_par = pd.DataFrame(par_time_means_arr)
df_median_par = pd.DataFrame(par_time_medians_arr)
df_mean_seq = pd.DataFrame(seq_time_means_arr)
df_median_seq = pd.DataFrame(seq_time_medians_arr)

df_mean_par.to_csv("pendulum_admm_means_par.csv")
df_median_par.to_csv("pendulum_admm_medians_par.csv")
df_mean_seq.to_csv("pendulum_admm_means_seq.csv")
df_median_seq.to_csv("pendulum_admm_medians_seq.csv")
