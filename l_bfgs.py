import numpy as np

from typing import Any, Tuple, Dict

import logging

class L_BFGS:

    def __init__(self, 
        m : int, 
        no_params : int, 
        gradient_func : Any
        ):
        self.m = m
        self.no_params = no_params
        self.params = np.random.rand(self.no_params)
        self.gradient_func = gradient_func

        # Updates on the params x over iterations
        self.s_arr = {}

        # Updates on the gradients g over iterations
        self.y_arr = {}

        # Store gradients
        self.gradients = {}

        # Logging
        handlerPrint = logging.StreamHandler()
        handlerPrint.setLevel(logging.DEBUG)
        self.log = logging.getLogger("l-bfgs")
        self.log.addHandler(handlerPrint)
        self.log.setLevel(logging.DEBUG)

    def calculate_update_first_step(self, q : np.array) -> np.array:
        H_k = np.eye(self.no_params)
        z = - np.dot(H_k, q)

        # Scale so initial magnitude is about 0.1 % of params
        factor = 0.001 * np.max(abs(self.params)) / np.max(abs(z))
        z *= factor

        return z

    def calculate_update(self, k : int, q : np.array) -> np.array:
        idx_start = max(k,0)
        idx_end = max(k-self.m,0)
        rho = {}
        alpha = {}
        self.log.debug("   Loop #1 range: %s" % list(reversed(range(idx_end, idx_start))))
        for i in reversed(range(idx_end, idx_start)):
            rho[i] = 1.0 / np.dot(self.y_arr[i], self.s_arr[i])
            alpha[i] = rho[i] * np.dot(self.s_arr[i], q)
            q = q - alpha[i] * self.y_arr[i]

        gamma_k = np.dot(self.s_arr[k-1], self.y_arr[k-1]) / np.dot(self.y_arr[k-1], self.y_arr[k-1])
        
        H_k_0 = gamma_k * np.eye(self.no_params)
        
        z = np.dot(H_k_0, q)

        idx_start = max(k-self.m,0)
        idx_end = k
        self.log.debug("   Loop #2 range: %s" % list(range(idx_start, idx_end)))
        for i in range(idx_start, idx_end):
            beta_i = rho[i] * np.dot(self.y_arr[i],z)
            z = z + self.s_arr[i] * (alpha[i] - beta_i)
        z = -z

        return z

    def get_gradient(self, 
        params : np.array, 
        gradient_noise_mag : float
        ) -> np.array:

        g = self.gradient_func(params)
        g += gradient_noise_mag * np.random.normal(size=self.no_params)
        return g

    def step(self, 
        k : int, 
        tol : float,
        gradient_noise_mag : float = 0.0,
        enforce_bounds : bool = False,
        bounds : np.array = np.array([]),
        tol_bounds : float = 1e-1
        ) -> Tuple[bool,np.array]:

        self.log.debug("Iteration: %d [start]" % k)

        # Reset if needed
        if k == 0:
            self.s_arr = {}
            self.y_arr = {}
            self.gradients = {}

        # Store gradient
        if not k in self.gradients:
            self.gradients[k] = self.get_gradient(self.params, gradient_noise_mag)
        q = self.gradients[k]

        # Calculate updates
        if k == 0:
            z = self.calculate_update_first_step(q)
        else:
            z = self.calculate_update(k, q)

        # New params
        params_new = self.params + z

        if enforce_bounds:
            assert len(bounds) == self.no_params

            for i in range(0,self.no_params):
                params_new[i] = max(bounds[i,0]+tol_bounds,params_new[i])
                params_new[i] = min(bounds[i,1]-tol_bounds,params_new[i])
            z = params_new - self.params

        self.log.debug("   Old params: %s" % self.params)
        self.log.debug("   New params: %s" % params_new)

        # Store change in params and gradients
        self.s_arr[k] = params_new - self.params
        self.log.debug("   S arr (change in params): %s" % self.s_arr[k])
        self.gradients[k+1] = self.get_gradient(params_new, gradient_noise_mag)
        self.y_arr[k] = self.gradients[k+1] - self.gradients[k]
        self.log.debug("   Y arr (change in grads): %s" % self.y_arr[k])

        # Remove old ones
        idx_remove = k - self.m -1
        if idx_remove > 0:
            del self.s_arr[idx_remove]
            del self.y_arr[idx_remove]
            del self.gradients[idx_remove]

        # Update params
        self.params = params_new
        
        self.log.debug("Iteration: %d [finished]" % k)

        # Monitor convergence
        if np.max(abs(z)) < tol:
            # if 0.5 * (np.max(abs(self.s_arr[k])) + np.max(abs(self.y_arr[k]))) < tol:
            return (True,z)
        else:
            return (False,z)

    def run(self, 
        no_steps : int, 
        params_init : np.array, 
        tol : float = 1e-8, 
        store_traj : bool = False,
        gradient_noise_mag : float = 0.0,
        enforce_bounds : bool = False,
        bounds : np.array = np.array([])
        ) -> Tuple[bool, int, np.array, Dict[int, np.array]]:

        assert no_steps >= 1

        self.params = params_init

        traj = {}
        if store_traj:
            traj[0] = self.params

        update = np.zeros(len(params_init))
        for k in range(0,no_steps):
            converged, update = self.step(
                k=k,
                tol=tol,
                gradient_noise_mag=gradient_noise_mag,
                enforce_bounds=enforce_bounds,
                bounds=bounds
                )
            
            if store_traj:
                traj[k+1] = self.params

            if converged:
                return (True, k, update, traj)
        
        return (False, no_steps, update, traj)