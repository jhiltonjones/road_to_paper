
import numpy as np
from scipy.optimize import linprog
from scipy.linalg import solve_discrete_are
from plot_code import overlay_mpc_logs
import scipy.sparse as sp
import osqp
from constants import *
from plot_code import print_rollout
from references import make_ref_sine
from scipy.linalg import block_diag
from beam_model_sim import load_interpolants_persisted  
from realtime_ref import RefStream, ScheduleEvent
import threading, time
import matplotlib.pyplot as plt
import time as wtime
from pid_controller import pid_compare_logged
theta_fn, J_fn = load_interpolants_persisted(cache_dir="cache")

def dare_stabilizing_K(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    K = -np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P
def trust_radius(jac_fn,
                 psi_rad,
                 h_rad=np.deg2rad(0.5),
                 eps_theta_rad=np.deg2rad(1.0),
                 Jmin=1e-6,
                 Lmin=1e-6,
                 dpsi_cap=np.deg2rad(5.0)):
    J0 = float(jac_fn(psi_rad))
    Jp = float(jac_fn(psi_rad + h_rad))
    Jm = float(jac_fn(psi_rad - h_rad))
    Jprime = (Jp - Jm) / (2.0*h_rad)

    dpsi_lin  = eps_theta_rad / max(abs(J0),    Jmin)
    dpsi_quad = (2.0*eps_theta_rad / max(abs(Jprime), Lmin))**0.5

    dpsi = min(dpsi_lin, dpsi_quad, dpsi_cap)
    return dpsi, {"J0": J0, "Jprime": Jprime, "dpsi_lin": dpsi_lin, "dpsi_quad": dpsi_quad}
def seq_mat_tv(Phi_list, B_list):
    N = len(Phi_list)
    n, m = B_list[0].shape
    Mx = np.zeros((N*n, n))
    Mc = np.zeros((N*n, N*m))

    # Prefix products P[i] = Φ_i ... Φ_0
    P = [None]*N
    P_prev = np.eye(n)
    for i in range(N):
        P_prev = Phi_list[i] @ P_prev
        P[i] = P_prev
        Mx[i*n:(i+1)*n, :] = P[i]

    # Mc blocks: T(i,j) = Φ_i ... Φ_{j+1} B_j  (empty product = I when i==j)
    for i in range(N):
        for j in range(i+1):
            if i == j:
                Tij = np.eye(n)
            else:
                Tij = np.eye(n)
                for s in range(i, j, -1):      # s = i, i-1, ..., j+1  (INCLUDES Φ_i)
                    Tij = Phi_list[s] @ Tij
            Mc[i*n:(i+1)*n, j*m:(j+1)*m] = Tij @ B_list[j]
    return Mx, Mc

def compute_VT_MPI(A, B, F, G, K=None, Q=None, R=None, tol=1e-8, nu_max=200):
    """
    Returns V_T and nu such that X_MPI = { x : V_T x <= 1 } with
    V_T = stack_{i=0..nu} (F + G K) Phi^i  and Phi = A + B K.
    Stops at the smallest nu that satisfies the theorem’s test.
    """
    n = A.shape[0]
    if K is None:
        if (Q is None) or (R is None):
            raise ValueError("Provide K, or Q and R to compute it.")
        K, _ = dare_stabilizing_K(A, B, Q, R)

    Phi = A + B @ K
    # M = F + G @ K           
    M = np.vstack([F, G @ K])

    # Start with V_0 = M
    V_stack = M.copy()
    Phi_pow = np.eye(n)

    for nu in range(0, nu_max+1):
        # check the condition for nu: (F+GK) Phi^{nu+1} x <= 1 for all x with V_stack x <= 1
        Phi_pow = Phi_pow @ Phi                  # Phi^{nu+1}
        W = M @ Phi_pow                          # rows to be verified

        ok_all = True
        for j in range(W.shape[0]):
            # LP: max   w_j^T x   s.t. V_stack x <= 1
            #     = min -w_j^T x
            c = -W[j]
            A_ub = V_stack
            b_ub = np.ones(V_stack.shape[0])
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, method="highs")
            if (res.status != 0):    
                raise RuntimeError("LP failed while checking MPI condition.")
            max_val = -res.fun
            if max_val > 1.0 + tol:
                ok_all = False
                break

        if ok_all:
            # nu found; build V_T = stack_{i=0..nu} M Phi^i
            # print(f"nu found at {nu}")
            V_T = []
            Phi_pow_i = np.eye(n)
            for i in range(nu+1):
                V_T.append(M @ Phi_pow_i)
                Phi_pow_i = Phi_pow_i @ Phi
            return np.vstack(V_T), nu

        # else: constraints at step nu+1 are not implied; add them and continue
        V_stack = np.vstack([V_stack, W])

    raise RuntimeError("Reached nu_max without satisfying the MPI condition.")

def build_nominal_B_list_with_bounds(psi_now, U_prev, N, dt, J_fn,
                                     trust_params=None, J_min=None, wrap=False):
    if trust_params is None:
        trust_params = dict(h_rad=np.deg2rad(0.5),
                            eps_theta_rad=np.deg2rad(1.0),
                            Jmin=1e-6, Lmin=1e-6,
                            dpsi_cap=np.deg2rad(5.0))

    # shift + hold-last nominal plan
    U_nom = np.zeros(N) if (U_prev is None) else np.r_[U_prev[1:], U_prev[-1]]

    # nominal ψ along horizon
    S = np.tril(np.ones((N, N), float)) * dt
    psi_nom = psi_now + S @ U_nom
    if wrap:
        psi_nom = (psi_nom + np.pi) % (2*np.pi) - np.pi

    B_list   = []
    dpsi_vec = np.zeros(N, float)
    for i in range(N):
        Ji = float(J_fn(psi_nom[i]))
        if J_min is not None:
            Ji = np.sign(Ji) * max(abs(Ji), J_min)   # sign-preserving floor
        B_list.append(np.array([[dt * Ji]]))         # (1×1) SISO

        dpsi_i, _ = trust_radius(J_fn, psi_nom[i],
                                 h_rad=trust_params["h_rad"],
                                 eps_theta_rad=trust_params["eps_theta_rad"],
                                 Jmin=trust_params["Jmin"],
                                 Lmin=trust_params["Lmin"],
                                 dpsi_cap=trust_params["dpsi_cap"])
        dpsi_vec[i] = dpsi_i

    return psi_nom, U_nom, B_list, dpsi_vec

def solve_qp_osqp(H, F, A, l, u, U_warm=None):
    P = sp.csc_matrix(0.5*(H + H.T))
    q = F.astype(float)
    A = sp.csc_matrix(A)
    prob = osqp.OSQP()
    prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, eps_abs=1e-6, eps_rel=1e-6)
    if U_warm is not None:
        prob.warm_start(x=U_warm)
    res = prob.solve()
    status = res.info.status 
    if status not in ("solved", "solved inaccurate"):
        return None, None, status
    return res.x, res.y, status  

def mpc_step_trust_region_prestab_terminal_TV_vK(
    A, Q, R, N, xk, xref_seq, dt, dpsi_max,
    *, K, V_T=None, psi_now=None, J_fn=None, U_prev=None,
    Qf=None, trust_params=None, J_min=None,
    idx_theta=0, error_threshold_deg=10.0
):
    n = A.shape[0]; m = 1

    # --- TV linearisation + local ψ-bounds ---
    psi_nom, U_nom, B_list, dpsi_vec = build_nominal_B_list_with_bounds(
        psi_now, U_prev, N, dt, J_fn,
        trust_params=trust_params, J_min=J_min, wrap=False
    )

    # backward Riccati with time-varying B_k
    P = Qf if Qf is not None else solve_discrete_are(A, B_list[-1], Q, R)
    K_seq = [None]*N
    P_next = P
    for k in range(N-1, -1, -1):
        Bk = B_list[k]
        S  = R + Bk.T @ P_next @ Bk
        Kk = -np.linalg.solve(S, Bk.T @ P_next @ A)
        K_seq[k] = Kk
        Acl = A + Bk @ Kk
        P_next = Q + A.T @ P_next @ Acl  # Riccati difference
        # print(f"K is {Kk} for the varying K")

    # closed-loop Φ_k and stacked matrices
    Phi_list = [A + B_list[k] @ K_seq[k] for k in range(N)]
    Mx, Mc = seq_mat_tv(Phi_list, B_list)

    # block-diagonal Kbar
    Kbar = block_diag(*K_seq)


    # Weights
    Qtil = np.zeros((N*n, N*n))
    if N > 1:
        Qtil[:(N-1)*n, :(N-1)*n] = np.kron(np.eye(N-1), Q)
    if Qf is None:
        Qf = solve_discrete_are(A, B_list[-1], Q, R)
    Qtil[(N-1)*n:, (N-1)*n:] = Qf
    Rtil = np.kron(np.eye(N), R)

    # Cost in c
    # Kbar = np.kron(np.eye(N), K)
    IUm  = np.eye(N*m)
    X0   = Mx @ xk
    KC   = Kbar @ Mc
    U0   = Kbar @ X0
 
    H = 2.0 * (Mc.T @ Qtil @ Mc + (KC + IUm).T @ Rtil @ (KC + IUm))
    f = 2.0 * (Mc.T @ Qtil @ (X0 - xref_seq) + (KC + IUm).T @ Rtil @ U0)
    H = 0.5 * (H + H.T)

    # --- Constraints ---
    A_osqp = np.empty((0, N*m)); l_osqp = np.empty(0); u_osqp = np.empty(0)

    # (A) ψ-linearisation tube:  |ψ_pred - ψ_nom| ≤ dpsi_bound
    # ψ_pred - ψ_nom = S*(U - U_nom) = S*( U0 - U_nom ) + S*(Kbar@Mc + I) c
    S = np.tril(np.ones((N, N), float)) * dt
    J_u   = Kbar @ Mc + IUm         # dU/dc
    A_tube = S @ J_u                # maps c → (ψ_pred - ψ_nom)
    offset = S @ (U0 - U_nom)       # nominal mismatch
    dpsi_bound = np.minimum(dpsi_vec, np.full(N, dpsi_max)) if (dpsi_max is not None) else dpsi_vec
    dpsi_bound = np.maximum(dpsi_bound, 1e-12)  # avoid zero-width

    l_tube = -dpsi_bound - offset
    u_tube = +dpsi_bound - offset

    A_osqp = np.vstack([A_osqp, A_tube])
    l_osqp = np.concatenate([l_osqp, l_tube])
    u_osqp = np.concatenate([u_osqp, u_tube])

    # (B) θ-band (unchanged)
    band = np.deg2rad(error_threshold_deg)
    e_theta = np.zeros((1, n)); e_theta[0, idx_theta] = 1.0
    E   = np.kron(np.eye(N), e_theta)
    EX0 = E @ X0
    theta_ref = np.asarray(xref_seq).reshape(N)
    A_th  = E @ Mc
    rhs_p = theta_ref + band - EX0
    rhs_n = -theta_ref + band + EX0

    A_osqp = np.vstack([A_osqp,  A_th,  -A_th])
    l_osqp = np.concatenate([l_osqp, -np.inf*np.ones(N), -np.inf*np.ones(N)])
    u_osqp = np.concatenate([u_osqp,  rhs_p,               rhs_n])

    # (C) Terminal set (optional)
    if V_T is not None:
        Sn = np.zeros((n, N*n)); Sn[:, (N-1)*n:N*n] = np.eye(n)
        A_term = V_T @ (Sn @ Mc)
        u_term = 1.0 - (V_T @ (Sn @ X0)).ravel()

        A_osqp = np.vstack([A_osqp, A_term])
        l_osqp = np.concatenate([l_osqp, -np.inf*np.ones_like(u_term)])
        u_osqp = np.concatenate([u_osqp,  u_term])

    # Solve
    c_opt, y, status = solve_qp_osqp(H, f, A_osqp, l_osqp, u_osqp)
    if status not in ("solved", "solved inaccurate") or (c_opt is None):
        return None, None, (K, None, None, H, f, Mx, Mc, Qtil, A_osqp, (l_osqp, u_osqp), y, status)

    X_pred = X0 + Mc @ c_opt
    U_pred = Kbar @ X_pred + c_opt
    return U_pred, X_pred, (K, None, None, H, f, Mx, Mc, Qtil, A_osqp, (l_osqp, u_osqp), y, status)


def mpc_compare_logged9(ref, T, dt, p_vec, *,
                       Np=15, w_th=1.0, w_u=1e-3,
                       psi0_rad=0.0, J_min=1e-5, s_steps=10,
                       eps_theta_deg=10.0, h_deg_for_radius=0.5, sigma_theta_meas_deg=0.2, error_threshold=10, theta_max = np.deg2rad(90), u_max = np.deg2rad(90), trust_region = np.deg2rad(10), step_cb=None):

    S = np.tril(np.ones((Np, Np), float)) * dt
    rng = np.random.default_rng(2)
    sigma_theta_meas = np.deg2rad(sigma_theta_meas_deg)

    log = dict(
        t=[], theta_ref_deg=[],

        u_free_deg_s=[], psi_tr_deg_cl=[],   u_tr_deg_s=[],   theta0_tr_deg_cl=[],   theta_pred_tr_k1_deg_cl=[],

        dpsi_max_deg=[], viol_tr_deg=[], du0_deg_s=[],
        
        theta_true_next_tr_deg_cl=[],
        e_traj_tr_deg_cl=[],
        se_traj_tr_deg2_cl=[],
    )

    psi_tr   = float(psi0_rad)

    N = int(round(T/dt))
    U_prev = np.zeros(Np)
    for k in range(N):
        t = k*dt
        A_th = np.array([[1.0]])
        Q_th = np.array([[w_th]]); R_u = np.array([[w_u]])
        # xref = np.array([ref(t + (i+1)*dt)[0] for i in range(Np)], float)
        xref = ref.sequence(t, Np, dt, lookahead=True, zoh=False)
        psi_k_tr = psi_tr
        theta0_tr_cl   = float(theta_fn(psi_k_tr))
        theta0_tr_n = float(theta0_tr_cl + rng.normal(0.0, sigma_theta_meas))
        J0_tr       = float(J_fn(psi_k_tr))
        B_th_tr = np.array([[max(abs(J0_tr), J_min) * np.sign(J0_tr) * dt]])        
        xk_tr = np.array([theta0_tr_n])

        dpsi_max, diag = trust_radius(lambda ψ: float(J_fn(ψ)),
                                    psi_k_tr,
                                    h_rad=np.deg2rad(h_deg_for_radius),
                                    eps_theta_rad=np.deg2rad(eps_theta_deg),
                                    Jmin=1e-6, Lmin=1e-6, dpsi_cap=np.deg2rad(180.0))

        if 'U_prev' not in locals():
            U_prev = np.zeros(Np)                  


        # State/input polytope: F x <= 1, G u <= 1
        F_state = np.array([[ 1.0/theta_max],
                            [-1.0/theta_max]])
        G_input = np.array([[ 1.0/u_max     ],
                            [-1.0/u_max     ]])

        K_dare, P_dare = dare_stabilizing_K(A_th, B_th_tr, Q_th, R_u)
        V_T, nu = compute_VT_MPI(A_th, B_th_tr, F_state, G_input, K=K_dare)  # keep M = vstack([F, G@K])
        # print(f"P is {P_dare}")
        # U_tr, X_tr, tr_pack = mpc_step_trust_region_prestab_terminal_TV(
        #     A_th, Q_th, R_u, Np, xk_tr, xref, dt, dpsi_max,
        #     K=K_dare, V_T=V_T,
        #     psi_now=psi_tr, J_fn=J_fn, U_prev=U_prev,
        #     Qf=P_dare,                        
        #     idx_theta=0, error_threshold_deg=error_threshold
        # )
        trust_params = dict(
            h_rad=np.deg2rad(h_deg_for_radius),
            eps_theta_rad=np.deg2rad(eps_theta_deg),
            Jmin=1e-6, Lmin=1e-6,
            dpsi_cap=trust_region  
        )

        U_tr, X_tr, tr_pack = mpc_step_trust_region_prestab_terminal_TV_vK(
            A_th, Q_th, R_u, Np, xk_tr, xref, dt, dpsi_max=np.deg2rad(180.0),  
            K=K_dare, V_T=V_T,
            psi_now=psi_tr, J_fn=J_fn, U_prev=U_prev,
            Qf=P_dare,
            trust_params=trust_params, J_min=J_min,          
            idx_theta=0, error_threshold_deg=error_threshold
        )
        
        status = tr_pack[-1]
        is_infeas = (status not in ("solved", "solved inaccurate"))
        log.setdefault("qp_status", []).append(1 if is_infeas else 0)


        # if is_infeas or (U_tr is None) or (X_tr is None):
        #     u0_tr = 0.0
        #     psi_tr_next = psi_tr
        #     U_prev = np.zeros(Np)
        #     U_tr_vec = np.zeros(Np)         
        # else:
        #     u0_tr = float(np.ravel(U_tr)[0])
        #     psi_tr_next = psi_tr + u0_tr*dt
        #     U_prev = np.asarray(U_tr, float).reshape(-1)
        #     U_tr_vec = U_prev                  

        if is_infeas or (U_tr is None) or (X_tr is None):
            u0_tr = 0.0
            psi_tr_next = psi_tr
            U_prev   = np.zeros(Np)
            U_tr_vec = np.zeros(Np)       
        else:
            u0_tr = float(np.ravel(U_tr)[0])
            psi_tr_next = psi_tr + u0_tr*dt
            U_prev   = np.asarray(U_tr, float).reshape(-1)
            U_tr_vec = U_prev           
        viol_tr    = float(np.max(np.abs(S @ U_tr_vec)))
        if X_tr is None:
            theta_pred_tr_k1_deg = np.nan
        else:
            theta_pred_tr_k1_deg = np.degrees(float(np.ravel(X_tr)[0]))

        theta_true_next_tr_cl   = float(theta_fn(psi_tr_next))
        # ---- diagnostics that never crash ----
        S_chk = np.tril(np.ones((Np, Np))) * dt
        dpsi  = S_chk @ U_tr_vec
        # print("u0*dt (deg):", np.degrees(U_tr_vec[0]*dt))
            # " <= dpsi_max (deg):", np.degrees(dpsi_max))
        # print("max|Δψ| (deg):", np.degrees(np.max(np.abs(dpsi))),
        #     "bound (deg):", np.degrees(dpsi_max))

        # optional: only pretty rollout when feasible
        if not is_infeas and (U_tr is not None) and (X_tr is not None):
            print_rollout(k, t, psi_tr, U_tr, X_tr, dt,
                        dpsi_max=dpsi_max,
                        theta_ref_seq=xref, idx_theta=0,
                        title="TV-MPC predicted rollout")

        log["t"].append(t)
        log["theta_ref_deg"].append(np.degrees(xref[0]))

        log["psi_tr_deg_cl"].append(np.degrees(psi_tr))
        log["u_tr_deg_s"].append(np.degrees(u0_tr))
        log["theta0_tr_deg_cl"].append(np.degrees(theta0_tr_cl))
        log["theta_pred_tr_k1_deg_cl"].append(theta_pred_tr_k1_deg)

        log["dpsi_max_deg"].append(np.degrees(dpsi_max))
        log["viol_tr_deg"].append(np.degrees(viol_tr))


        log["theta_true_next_tr_deg_cl"].append(np.degrees(theta_true_next_tr_cl))
        e_tr   = log["theta_true_next_tr_deg_cl"][-1]   - log["theta_ref_deg"][-1]

        log["e_traj_tr_deg_cl"].append(e_tr)
        log["se_traj_tr_deg2_cl"].append(e_tr**2)
        if step_cb is not None:
            step_cb(k, t, log)

        psi_tr   = psi_tr_next

    for k in list(log.keys()):
        if k not in ("Np","dt"):
            log[k] = np.asarray(log[k], float)

    log["rmse_traj_tr_deg_cl"]   = float(np.sqrt(np.mean(log["se_traj_tr_deg2_cl"])))   if log["se_traj_tr_deg2_cl"].size   else np.nan

    log["Np"] = Np; log["dt"] = dt
    return log



if __name__ == "__main__":


    p_vec = np.array([LENGTH, 0.18, 0.0])
    T_total = 50.0
    dt = 0.1
    Np = 5
    w_th, w_u = 10.0, 0.05
    psi0 = np.deg2rad(0.0)

    # --- sine RefStream in RADIANS ---
    A = np.deg2rad(15.0)     # amplitude
    f = 0.2                  # Hz
    off = 0.0
    ref = RefStream(func=lambda t: off + A*np.sin(2*np.pi*f*t))

    # --- live figure ---
    plt.ion()
    fig, ax = plt.subplots(2, 1, figsize=(9,6), sharex=True)

    ln_ref,  = ax[0].plot([], [], label="θ_ref [deg]")
    ln_true, = ax[0].plot([], [], label="θ_true [deg]")
    ax[0].grid(True); ax[0].legend(loc="upper right"); ax[0].set_ylabel("deg")

    ln_u,    = ax[1].plot([], [], label="u [deg/s]")
    ln_psi,  = ax[1].plot([], [], label="ψ [deg]")
    ax[1].grid(True); ax[1].legend(loc="upper right")
    ax[1].set_ylabel("deg / deg/s"); ax[1].set_xlabel("time [s]")

    # histories (read from the log dict in the callback)
    times = []
    theta_ref_deg_hist   = []
    theta_true_deg_hist  = []
    psi_deg_hist         = []
    u_deg_s_hist         = []

    t0_wall = wtime.time()
    def step_cb_mpc(k, t, log):
        """Called by mpc_compare_logged9 each step to update the plot and pace wall-clock time."""
        # extend histories from the last appended items in log
        times.append(log["t"][-1])
        theta_ref_deg_hist.append(log["theta_ref_deg"][-1])
        theta_true_deg_hist.append(log["theta_true_next_tr_deg_cl"][-1])
        psi_deg_hist.append(log["psi_tr_deg_cl"][-1])
        u_deg_s_hist.append(log["u_tr_deg_s"][-1])

        # update lines
        ln_ref.set_data(times, theta_ref_deg_hist)
        ln_true.set_data(times, theta_true_deg_hist)
        ln_u.set_data(times, u_deg_s_hist)
        ln_psi.set_data(times, psi_deg_hist)

        # rescale and draw
        for axx in ax:
            axx.relim(); axx.autoscale_view()
        plt.pause(0.001)

        # pace to real time
        wtime.sleep(max(0.0, (t0_wall + (k+1)*dt) - wtime.time()))
    def step_cb_pid(k, t, log):
        """Called by mpc_compare_logged9 each step to update the plot and pace wall-clock time."""
        # extend histories from the last appended items in log
        times.append(log["t"][-1])
        theta_ref_deg_hist.append(log["theta_ref_deg"][-1])
        theta_true_deg_hist.append(log["theta_true_next_tr_deg_cl"][-1])
        psi_deg_hist.append(log["psi_tr_deg_cl"][-1])
        u_deg_s_hist.append(log["u_tr_deg_s"][-1])

        # update lines
        ln_ref.set_data(times, theta_ref_deg_hist)
        ln_true.set_data(times, theta_true_deg_hist)
        ln_u.set_data(times, u_deg_s_hist)
        ln_psi.set_data(times, psi_deg_hist)

        # rescale and draw
        for axx in ax:
            axx.relim(); axx.autoscale_view()
        plt.pause(0.001)

        # pace to real time
        wtime.sleep(max(0.0, (t0_wall + (k+1)*dt) - wtime.time()))

    # --- run your MPC with live plotting via callback ---
    log_mpc = mpc_compare_logged9(
        ref=ref,
        T=T_total,
        dt=dt,
        p_vec=p_vec,
        Np=Np,
        w_th=w_th,
        w_u=w_u,
        psi0_rad=psi0,
        J_min=1e-5,
        s_steps=10,
        eps_theta_deg=10.0,
        sigma_theta_meas_deg=5.0,
        error_threshold=np.inf,
        trust_region=np.deg2rad(180),
        step_cb=None,
    )

    # reset live-plot histories before PID (or open a new figure)
    times.clear(); theta_ref_deg_hist.clear(); theta_true_deg_hist.clear()
    psi_deg_hist.clear(); u_deg_s_hist.clear()
    t0_wall = wtime.time()  # reset pacing origin

    # --- run PID (optional live plot) ---
    log_pid = pid_compare_logged(
        ref=ref, T=T_total, dt=dt, theta_fn=theta_fn, J_fn=J_fn,
        psi0_rad=psi0,
        Kp=2.0,
        sigma_theta_meas_deg=5.0,
        step_cb=step_cb_pid
    )

    # --- overlay final results ---
    overlay_mpc_logs(
        [log_mpc, log_pid],
        label_list=["MPC (TV, vK)", "PID (θ-loop)"],
        color_list=["tab:blue", "tab:orange"]
    )

    # keep window open
    plt.ioff()
    plt.show()
