import jax
import jax.numpy as jnp
import pickle
from cparcon.utils import euler, rollout
from jax import config
from cparcon.optimal_control_problem import OCP
from cparcon.par_ip_newton import par_interior_point_optimal_control
from jax import debug

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

# select platform
config.update("jax_platform_name", "cpu")

with open("IdentificationResults.pkl", "rb") as f:
    id_results = pickle.load(f)

# Clumsy way to select the object corresponding to the final concentrate grade
FC_Au = [
    x
    for x in id_results
    if x["cv"] == "Ada Tepe dynamic/Final Conc/Weight percent (species) Au"
][0]

A = jnp.array(FC_Au["A"])
B = jnp.array(FC_Au["B"])
C = jnp.array(FC_Au["C"])


control_lb = jnp.array(
    [
        80.0,
        50.0,
        2.0,
        175.0,
        200.0,
        0.1,
        0.1,
        0.1,
        0.1,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
)


control_ub = jnp.array(
    [
        120.0,  # Ada Tepe dynamic/DataMappings/Custom/_Signal_FEED_RATE_TPH
        75.0,  # Ada Tepe dynamic/DataMappings/Custom/_Signal_FEED_SIZE_P80
        8.0,  # Ada Tepe dynamic/DataMappings/Custom/_Signal_FEED_Au_ppm_SP
        300.0,  # Ada Tepe dynamic/DataMappings/Custom/_Signal_FeedFlow_avg
        500.0,  # Ada Tepe dynamic/DataMappings/Custom/_Signal_PAXDosingSP
        25.0,  # Ada Tepe dynamic/TK-001/Custom/_Signal_ADAPT_Ro_Qtz
        25.0,  # Ada Tepe dynamic/TK-001/Custom/_Signal_ADAPT_Ro_k_Au
        25.0,  # Ada Tepe dynamic/kinetic/Custom/_Signal_ADAPT_Scav_Qtz
        25.0,  # Ada Tepe dynamic/TK-002/Custom/_Signal_ADAPT_Clnr_Qtz
        23.0,
        42.0,
        53.0,
        54.0,
        60.0,
        50.0,
        51.0,
        49.0,
        15.0,
        16.0,
        42.0,
        42.0,
        42.0,
        4.0,
        42.0,
        16.0,
        16.0,
        16.0,
        17.0,
    ]
)

base_control = control_lb + (control_ub - control_lb) / 2.0

def linear_model(state, control):
    return A @ state + B @ control


def observation_model(state):
    return C @ state + 19600

sampling_period = 5
dynamics = euler(linear_model, sampling_period)
horizon = 10000
control_sequence = jnp.kron(base_control, jnp.ones((horizon, 1)))
initial_state = jnp.zeros(len(FC_Au["A"]))
states = rollout(dynamics, control_sequence, initial_state)
initial_state = states[-1]

def constraints(state, control):
    c0 = control - control_ub
    c1 = -control + control_lb
    return jnp.hstack((c0, c1))

def final_cost(state):
    goal_state = 450.
    err = goal_state - observation_model(state)
    final_state_cost = jnp.array([[1e0]])
    c = 0.5 * err.T @ final_state_cost @ err
    return c

def stage_cost(state, action, bp):
    goal_state = 450.
    err = goal_state - observation_model(state)
    state_cost = jnp.array([[1e-3]])
    action_cost = 1e-1*jnp.diag(jnp.array([
        1e0,
        1e0,
        1e0,
        1e0,
        1e0,
        1e0,
        1e0,
        1e0,
        1e0,
        1e0,
        1e0,
        1e0,
        1e0,
        1e0,
        1e0,
        1e0,
        1e0,
        1e0,
        1e0,
        1e0,
        1e0,
        1e0,
        1e0,
        1e0,
        1e0,
        1e0,
        1e0,
        1e0,
    ]))
    c = 0.5 * err.T @ state_cost @ err
    c += 0.5 * action.T @ action_cost @ action
    log_barrier = jnp.sum(jnp.log(-constraints(state, action)))
    return c - bp * log_barrier

def total_cost(states: jnp.ndarray, controls: jnp.ndarray, bp: float):
    ct = jax.vmap(stage_cost, in_axes=(0, 0, None))(states[:-1], controls, bp)
    cT = final_cost(states[-1])
    return cT + jnp.sum(ct)

N = 12
control_sequence = jnp.kron(base_control, jnp.ones((N, 1)))
ocp = OCP(dynamics, constraints, stage_cost, final_cost, total_cost)
optimal_controls, _ = par_interior_point_optimal_control(ocp, control_sequence, initial_state)
states = rollout(dynamics, optimal_controls, initial_state)
outputs = jax.vmap(observation_model)(states)
