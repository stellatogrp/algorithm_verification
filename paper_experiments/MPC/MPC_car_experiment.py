from datetime import datetime

import numpy as np
import pandas as pd
from car2D import Car2D
from tqdm import trange


def single_sim(car, T_sim, xinit, uinit, eps=1e-3):
    A, B = car.A, car.B
    for _ in range(T_sim):
        sol = car.solve_via_cvxpy(xinit, uinit=uinit)
        uinit = sol[car.nx: car.nx+car.nv]
        xinit = A @ xinit + B @ uinit + eps * np.random.normal(size=(car.nx,))
        # print(xinit, uinit)

    return xinit, uinit, sol


def simulate_steps(T=5, T_sim=25, N=100, eps=1e-3):
    np.random.seed(2)
    car = Car2D(T=T)
    xinit = np.array([5, 5, 0, 0])
    uinit = np.array([0, 0])
    # sol = car.solve_via_cvxpy(xinit)
    # u1 = sol[car.nx:car.nx+car.nv]
    # A, B = car.A, car.B

    # for _ in range(T_sim):
    #     sol = car.solve_via_cvxpy(xinit, uinit=uinit)
    #     uinit = sol[car.nx: car.nx+car.nv]
    #     xinit = A @ xinit + B @ uinit + eps * np.random.normal(size=(car.nx,))
    #     print(xinit, uinit)
    shifted_sols = []
    xinit_samples = []
    uinit_samples = []
    for _ in trange(N, desc='sampling'):
        out_xinit, out_uinit, sol = single_sim(car, T_sim, xinit, uinit, eps=eps)
        shifted_sols.append(shift_sol(sol, car))
        xinit_samples.append(out_xinit)
        uinit_samples.append(out_uinit)
    # print(np.array(samples))
    # print(np.min(xinit_samples, axis=0))
    # print(np.max(xinit_samples, axis=0))
    # print(np.min(uinit_samples, axis=0))
    # print(np.max(uinit_samples, axis=0))

    return car, xinit_samples, uinit_samples, sol, shifted_sols


def shift_sol(sol, car):
    out = np.zeros(sol.shape)
    out[:car.nx] = sol[:car.nx].copy()
    out[car.nx:car.nx + (car.T-2) * car.nv] = sol[car.nx + car.nv:]
    A, B = car.A, car.B
    # print(A @ sol[:car.nx] + B @ sol[car.nx: car.nx + car.nv])
    out[:car.nx] = A @ sol[:car.nx] + B @ sol[car.nx: car.nx + car.nv]
    # print(sol)
    # print('shifted ws:', out)
    return out

def MPC_experiment(outf, K_min=5, K_max=7, eps=1e-2):
    T = 5
    car, xinit_samples, uinit_samples, sol, shifted_sols = simulate_steps(T=T, N=100, eps=eps)
    xinit_min = np.min(xinit_samples, axis=0)
    xinit_max = np.max(xinit_samples, axis=0)
    uinit_min = np.min(uinit_samples, axis=0)
    uinit_max = np.max(uinit_samples, axis=0)
    # K = 2
    print(xinit_min, xinit_max)
    print(uinit_min, uinit_max)

    x0_min = np.min(shifted_sols, axis=0)
    x0_max = np.max(shifted_sols, axis=0)
    print(x0_min, x0_max)

    # options

    # for K in range(K_max):
    ws_x_val = shift_sol(sol, car)
    # experiments = [('cs', 'rho_const'), ('cs', 'rho_adj'), ('ws', 'rho_const'), ('ws', 'rho_adj')]
    # experiments = [('cs', 'rho_const'), ('cs', 'rho_adj')]
    # experiments = [('ws', 'rho_const'), ('ws', 'rho_adj')]
    # experiments = [('cs', 'rho_const')]
    # experiments = [('cs', 'rho_adj')]
    # experiments = [('ws', 'rho_const')]
    experiments = [('ws', 'rho_adj')]

    res = []
    for (start, rho) in experiments:
        for K in range(K_min, K_max + 1):
        # for K in range(6, 7):
            print(start, rho, K)
            if start == 'cs':
                ws_x = None
                shifted_sol_list = None
            else:
                ws_x = ws_x_val
                shifted_sol_list = shifted_sols
            if rho == 'rho_const':
                rho_const = True
            else:
                rho_const = False

            CP = car.get_CP(K, xinit_min, xinit_max, uinit_min, uinit_max, rho_const=rho_const,
                            ws_x=ws_x, shifted_sols=shifted_sol_list)
            # out = CP.solve(solver_type='SDP_CUSTOM')
            out = CP.solve(solver_type='GLOBAL', add_bounds=True, TimeLimit=3600)
            out['seed'] = 0
            out['start'] = start
            out['rho'] = rho
            out['T'] = T
            out['K'] = K

            res.append(pd.Series(out))
            res_df = pd.DataFrame(res)
            print(res_df)
            res_df.to_csv(outf, index=False)

    # CP = car.get_CP(K, xinit_min, xinit_max, uinit_min, uinit_max, rho_const=False, ws_x=ws_x)
    # out = CP.solve(solver_type='SDP_CUSTOM')
    # print(out)


def main():
    d = datetime.now()
    # print(d)
    curr_time = d.strftime('%m%d%y_%H%M%S')
    outf_prefix = '/home/vranjan/algorithm-certification/'
    # outf_prefix = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/'
    outf = outf_prefix + f'paper_experiments/MPC/data/{curr_time}.csv'
    print(outf)

    # simulate_steps()
    MPC_experiment(outf)


if __name__ == '__main__':
    main()
