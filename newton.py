import numpy as np

from typing import Any, Tuple, Dict

import logging

class Newton:

    def __init__(self, 
        no_params : int, 
        gradient_func : Any,
        inv_hessian_func : Any,
        lr : float
        ):
        self.no_params = no_params
        self.params = np.random.rand(self.no_params)
        self.gradient_func = gradient_func
        self.inv_hessian_func = inv_hessian_func
        self.lr = lr

        # Logging
        handlerPrint = logging.StreamHandler()
        handlerPrint.setLevel(logging.DEBUG)
        self.log = logging.getLogger("l-bfgs")
        self.log.addHandler(handlerPrint)
        self.log.setLevel(logging.DEBUG)

    def calculate_update(self,
        params : np.array, 
        gradient_noise_mag : float
        ) -> np.array:

        inv_hessian = self.inv_hessian_func(params)
        gradient = self.get_gradient(params, gradient_noise_mag)
        return - np.dot(inv_hessian, gradient)

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

        z = self.lr * self.calculate_update(self.params, gradient_noise_mag)

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

        # Update params
        self.params = params_new

        self.log.debug("Iteration: %d [finished]" % k)

        # Monitor convergence
        if np.max(abs(z)) < tol:
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