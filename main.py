import logging
from l_bfgs import L_BFGS
import plot
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

def obj_func(x : np.array) -> float:
    # Six hump function
    # http://scipy-lectures.org/intro/scipy/auto_examples/plot_2d_minimization.html
    return ((4 - 2.1*x[0]**2 + x[0]**4 / 3.) * x[0]**2 + x[0] * x[1] + (-4 + 4*x[1]**2) * x[1] **2)

def gradient(x : np.array) -> np.array:
    return np.array([
        8*x[0] - 4 * 2.1 * x[0]**3 + 2 * x[0]**5 + x[1],
        x[0] - 8 * x[1] + 16 * x[1]**3
        ])

def inv_hessian(x : np.array) -> np.array:
    return np.linalg.inv(np.array([
        [
            2 * (4 - 6 * 2.1 * x[0]**2 + 5 * x[0]**4),
            1
        ],
        [
            1,
            -8 + 48 * x[1]**2
        ]]))

def get_random_uniform_in_range(x_rng : np.array, y_rng : np.array) -> np.array:
    p = np.random.rand(2)
        
    p[0] *= x_rng[1] - x_rng[0]
    p[0] += x_rng[0]

    p[1] *= y_rng[1] - y_rng[0]
    p[1] += y_rng[0]

    return p

if __name__ == "__main__":

    opt = L_BFGS(
        m=5,
        no_params=2,
        gradient_func=gradient
    )
    opt.log.setLevel(logging.INFO)

    x_rng = [-2,2]
    y_rng = [-1,1]

    # Noise
    gradient_noise_mag = 0.0
    
    fig_dir = "figures"
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    title_desc = "gradient noise magnitude: %f" % gradient_noise_mag
    fname_ext = "%.4f" % gradient_noise_mag

    trajs = {}
    for trial in range(0,100):
        
        converged, no_steps_to_convergence, final_update, traj = opt.run(
            no_steps=500,
            params_init=get_random_uniform_in_range(x_rng, y_rng),
            store_traj=True,
            gradient_noise_mag=gradient_noise_mag,
            tol=1e-8,
            enforce_bounds=True,
            bounds=np.array([x_rng,y_rng])
        )

        # if converged:
        trajs[trial] = traj

        opt.log.info("Trial: %d converged: %s no steps: %d final params: %s final update: %s" % (trial,converged,no_steps_to_convergence,opt.params,final_update))

    endpoints, counts = plot.get_endpoints_and_counts(trajs)
    print("--- Endpoints ---")
    for i in range(0,len(endpoints)):
        print(endpoints[i], " : ", counts[i])
    
    print(trajs[1])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot.plot_3d_endpoint_lines(ax, trajs)
    plot.plot_3d(ax, x_rng,y_rng,obj_func)
    plt.savefig(fig_dir+"/3d.png", dpi=200)

    plt.figure()
    plot.plot_obj_func(x_rng, y_rng, obj_func)
    plot.plot_trajs(x_rng, y_rng, trajs, [1,2,3])
    plt.title("Trajs: " + title_desc)
    plt.savefig(fig_dir+"/trajs_%s.png" % fname_ext, dpi=200)

    plt.figure()
    plot.plot_obj_func(x_rng, y_rng, obj_func)
    # plot.plot_quiver(x_rng, y_rng, trajs)
    plot.plot_endpoint_counts(trajs)
    plt.title("Endpoints: " + title_desc)
    plt.savefig(fig_dir+"/endpoints_%s.png" % fname_ext, dpi=200)

    plt.figure()
    plot.plot_histogram(x_rng,y_rng,trajs,50)
    plot.plot_endpoint_counts(trajs)
    plt.title("Endpoints: " + title_desc)
    plt.savefig(fig_dir+"/histogram_%s.png" % fname_ext, dpi=200)

    # plt.show()


