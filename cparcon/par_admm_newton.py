import jax.numpy as jnp
from jax import grad, hessian, jacrev
from jax import vmap, debug
from jax import lax
from cparcon.optimal_control_problem import ADMM_OCP, Derivatives
from paroc.lqt_problem import LQT
from paroc import par_bwd_pass, par_fwd_pass
from typing import Callable
from cparcon.costates import par_costates


def compute_derivatives(
    ocp: ADMM_OCP,
    states: jnp.ndarray,
    controls: jnp.ndarray,
    consensus: jnp.ndarray,
    dual: jnp.ndarray,
):
    def body(x, u, z, l):
        cx_k, cu_k = grad(ocp.stage_cost, (0, 1))(x, u, z, l)
        cxx_k = hessian(ocp.stage_cost, 0)(x, u, z, l)
        cuu_k = hessian(ocp.stage_cost, 1)(x, u, z, l)
        cxu_k = jacrev(jacrev(ocp.stage_cost, 0), 1)(x, u, z, l)
        fx_k = jacrev(ocp.dynamics, 0)(x, u)
        fu_k = jacrev(ocp.dynamics, 1)(x, u)
        fxx_k = jacrev(jacrev(ocp.dynamics, 0), 0)(x, u)
        fuu_k = jacrev(jacrev(ocp.dynamics, 1), 1)(x, u)
        fxu_k = jacrev(jacrev(ocp.dynamics, 0), 1)(x, u)
        return cx_k, cu_k, cxx_k, cuu_k, cxu_k, fx_k, fu_k, fxx_k, fuu_k, fxu_k

    cx, cu, cxx, cuu, cxu, fx, fu, fxx, fuu, fxu = vmap(body)(
        states[:-1], controls, consensus, dual
    )
    return Derivatives(cx, cu, cxx, cuu, cxu, fx, fu, fxx, fuu, fxu)


def compute_lqr_params(lagrange_multipliers: jnp.ndarray, d: Derivatives):
    def body(l, cu, cxx, cuu, cxu, fu, fxx, fuu, fxu):
        r = cu + fu.T @ l
        Q = cxx + jnp.tensordot(l, fxx, axes=1)
        R = cuu + jnp.tensordot(l, fuu, axes=1)
        M = cxu + jnp.tensordot(l, fxu, axes=1)
        return r, Q, R, M

    return vmap(body)(
        lagrange_multipliers[1:], d.cu, d.cxx, d.cuu, d.cxu, d.fu, d.fxx, d.fuu, d.fxu
    )


def noc_to_lqt(
    ru: jnp.ndarray,
    Q: jnp.ndarray,
    R: jnp.ndarray,
    M: jnp.ndarray,
    A: jnp.ndarray,
    B: jnp.ndarray,
):
    T = Q.shape[0]
    nx = Q.shape[1]
    nu = R.shape[1]

    def offsets(Xt, Ut, Mt, rut):
        XiM = jnp.linalg.solve(Xt, Mt)
        st = -jnp.linalg.solve(Ut - Mt.T @ XiM, rut)
        rt = -XiM @ st
        return rt, st

    r, s = vmap(offsets)(Q, R, M, ru)
    H = jnp.eye(nx)
    HT = H
    H = jnp.kron(jnp.ones((T, 1, 1)), H)
    Z = jnp.eye(nu)
    Z = jnp.kron(jnp.ones((T, 1, 1)), Z)
    XT = Q[0]
    rT = jnp.zeros(nx)
    c = jnp.zeros((T, nx))
    lqt = LQT(A, B, c, XT, HT, rT, Q, H, r, R, Z, s, M)
    return lqt


def par_solution(
    nominal_states: jnp.ndarray,
    derivatives: Derivatives,
    reg_param: float,
    ru: jnp.ndarray,
    Q: jnp.ndarray,
    R: jnp.ndarray,
    M: jnp.ndarray,
):
    R = R + jnp.kron(jnp.ones((R.shape[0], 1, 1)), reg_param * jnp.eye(R.shape[1]))
    lqt = noc_to_lqt(ru, Q, R, M, derivatives.fx, derivatives.fu)
    Kx_par, d_par, S_par, v_par, pred_reduction, convex_problem = par_bwd_pass(lqt)
    du_par, dx_par = par_fwd_pass(
        lqt, jnp.zeros(nominal_states[0].shape[0]), Kx_par, d_par
    )
    return dx_par, du_par, pred_reduction, convex_problem, ru


def argmin_xu(
    ocp: ADMM_OCP,
    states: jnp.ndarray,
    controls: jnp.ndarray,
    consensus: jnp.ndarray,
    dual: jnp.ndarray,
):
    mu0 = 1.0
    nu0 = 2.0

    def while_body(val):
        x, u, z, l, iteration_counter, reg_param, reg_inc, _ = val
        cost = ocp.total_cost(x, u, z, l)
        derivatives = compute_derivatives(ocp, x, u, z, l)
        costates = par_costates(ocp, x[-1], derivatives)
        ru, Q, R, M = compute_lqr_params(costates, derivatives)

        def while_inner_loop(inner_val):
            _, _, _, _, rp, r_inc, inner_it_counter = inner_val
            dx, du, predicted_reduction, bwd_pass_feasible, Hu = par_solution(
                x, derivatives, rp, ru, Q, R, M
            )
            temp_u = u + du
            temp_x = x + dx
            Hu_norm = jnp.max(jnp.abs(Hu))
            new_cost = ocp.total_cost(temp_x, temp_u, z, l)
            actual_reduction = new_cost - cost
            gain_ratio = actual_reduction / predicted_reduction
            succesful_minimzation = jnp.logical_and(gain_ratio > 0.0, bwd_pass_feasible)
            rp = jnp.where(
                succesful_minimzation,
                rp * jnp.maximum(1.0 / 3.0, 1.0 - (2.0 * gain_ratio - 1.0) ** 3),
                rp * r_inc,
            )
            r_inc = jnp.where(succesful_minimzation, 2.0, 2 * r_inc)
            rp = jnp.clip(rp, 1e-16, 1e16)
            inner_it_counter += 1
            return (
                temp_x,
                temp_u,
                succesful_minimzation,
                Hu_norm,
                rp,
                r_inc,
                inner_it_counter,
            )

        def while_inner_cond(inner_val):
            _, _, succesful_minimzation, _, _, _, inner_it_counter = inner_val
            exit_cond = jnp.logical_or(succesful_minimzation, inner_it_counter > 500)
            return jnp.logical_not(exit_cond)

        x, u, _, Hamiltonian_norm, reg_param, reg_inc, _ = lax.while_loop(
            while_inner_cond,
            while_inner_loop,
            (x, u, jnp.bool_(0.0), 0.0, reg_param, reg_inc, 0),
        )
        iteration_counter = iteration_counter + 1
        return x, u, z, l, iteration_counter, reg_param, reg_inc, Hamiltonian_norm

    def while_cond(val):
        _, _, _, _, iteration_counter, _, _, Hu_norm = val
        exit_cond = jnp.logical_or(Hu_norm < 1e-4, iteration_counter > 500)
        return jnp.logical_not(exit_cond)

    opt_x, opt_u, _, _, iterations, _, _, _ = lax.while_loop(
        while_cond,
        while_body,
        (states, controls, consensus, dual, 0, mu0, nu0, jnp.array(1.0)),
    )
    return opt_x, opt_u, iterations


def argmin_z(
    ocp: ADMM_OCP, states: jnp.ndarray, controls: jnp.ndarray, dual: jnp.ndarray
):
    z = jnp.hstack((states[:-1], controls)) + 1 / ocp.penalty_parameter * dual
    return vmap(ocp.projection)(z)


def grad_ascent(
    ocp: ADMM_OCP,
    states: jnp.ndarray,
    controls: jnp.ndarray,
    consensus: jnp.ndarray,
    dual: jnp.ndarray,
):
    return dual + ocp.penalty_parameter * (
        jnp.hstack((states[:-1], controls)) - consensus
    )


def primal_residual(states: jnp.ndarray, controls: jnp.ndarray, dual: jnp.ndarray):
    return jnp.max(jnp.abs(jnp.hstack((states[:-1], controls)) - dual))


def par_admm(
    stage_cost: Callable,
    final_cost: Callable,
    dynamics: Callable,
    projection: Callable,
    states0: jnp.ndarray,
    controls0: jnp.ndarray,
    consensus0: jnp.ndarray,
    dual0: jnp.ndarray,
    penalty_param: float,
):
    def admm_stage_cost(x, u, z, l):
        y = jnp.hstack((x, u))
        sql2norm = (y - z + 1.0 / penalty_param * l).T @ (
            y - z + 1.0 / penalty_param * l
        )
        return stage_cost(x, u) + penalty_param / 2 * sql2norm

    def admm_total_cost(states, controls, consensus, dual):
        ct = vmap(admm_stage_cost)(states[:-1], controls, consensus, dual)
        cT = final_cost(states[-1])
        return cT + jnp.sum(ct)

    admm_ocp = ADMM_OCP(
        dynamics,
        projection,
        admm_stage_cost,
        final_cost,
        admm_total_cost,
        penalty_param,
    )

    def admm_iteration(val):
        x, u, z, l, _, _, it_cnt = val
        x, u, it = argmin_xu(admm_ocp, x, u, z, l)
        it_cnt += it
        prev_z = z
        z = argmin_z(admm_ocp, x, u, l)

        l = grad_ascent(admm_ocp, x, u, z, l)

        rp_infty = primal_residual(x, u, z)
        rd_infty = jnp.max(jnp.abs(z - prev_z))
        it_cnt += 1
        return x, u, z, l, rp_infty, rd_infty, it_cnt

    def admm_conv(val):
        _, _, _, _, rp_infty, rd_infty, _ = val
        exit_condition = jnp.logical_and(rp_infty < 1e-2, rd_infty < 1e-2)
        return jnp.logical_not(exit_condition)

    (
        opt_states,
        opt_controls,
        opt_consensus,
        opt_dual,
        _,
        _,
        iterations,
    ) = lax.while_loop(
        admm_conv,
        admm_iteration,
        (states0, controls0, consensus0, dual0, jnp.inf, jnp.inf, 0.0),
    )
    # debug.print("iterations      {x}", x=iterations)
    # debug.print("------------------------------")
    # debug.breakpoint()
    return opt_states, opt_controls, opt_consensus, opt_dual, iterations
