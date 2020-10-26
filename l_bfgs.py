import numpy as np

from typing import Any, Tuple, Dict

import logging

class NotDescentDirection(Exception):
    pass

class ZeroDescentProduct(Exception):
    pass

class ZeroUpdate(Exception):
    pass

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
        self.gradients_last = None
        self.params_last = None

        # Logging
        handlerPrint = logging.StreamHandler()
        handlerPrint.setLevel(logging.DEBUG)
        self.log = logging.getLogger("l-bfgs")
        self.log.addHandler(handlerPrint)
        self.log.setLevel(logging.DEBUG)

    def calculate_update_first_step(self, q : np.array) -> np.array:
        return - 1e-2 * q # Small!
        
    def get_y_arr_reg(self, i : int, eps_reg : float) -> np.array:
        return self.y_arr[i] + eps_reg * self.s_arr[i]

    def calculate_update(self, k : int, q : np.array, eps_reg : float) -> np.array:
        idx_start = max(k,0)
        idx_end = max(k-self.m,0)
        rho = {}
        alpha = {}
        # self.log.debug("   Loop #1 range: %s" % list(reversed(range(idx_end, idx_start))))
        for i in reversed(range(idx_end, idx_start)):
            rho[i] = 1.0 / np.dot(self.get_y_arr_reg(i,eps_reg), self.s_arr[i])
            alpha[i] = rho[i] * np.dot(self.s_arr[i], q)
            q = q - alpha[i] * self.get_y_arr_reg(i,eps_reg)

        gamma_k = np.dot(self.s_arr[k-1], self.get_y_arr_reg(k-1,eps_reg)) \
            / np.dot(self.get_y_arr_reg(k-1,eps_reg), self.get_y_arr_reg(k-1,eps_reg))
        
        H_k_0 = gamma_k * np.eye(self.no_params)
        
        z = np.dot(H_k_0, q)

        idx_start = max(k-self.m,0)
        idx_end = k
        # self.log.debug("   Loop #2 range: %s" % list(range(idx_start, idx_end)))
        for i in range(idx_start, idx_end):
            beta_i = rho[i] * np.dot(self.get_y_arr_reg(i,eps_reg),z)
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

    def get_descent_inner_product(self,
        p : np.array,
        gradients : np.array
        ) -> float:
        inner_prod = np.dot(p, gradients)

        if inner_prod > -1e-16 and inner_prod <= 0:
            raise ZeroDescentProduct()
        elif inner_prod > 0:
            raise NotDescentDirection()
        
        return inner_prod
    
    def check_is_descent_direction(self,
        p : np.array,
        gradients : np.array
        ) -> bool:
        try:
            self.get_descent_inner_product(p,gradients)
            return True
        except NotDescentDirection:
            return False

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

        # Store gradient
        gradients = self.get_gradient(self.params, gradient_noise_mag)

        # Calculate updates
        if k == 0:
            update = self.calculate_update_first_step(gradients)
        else:

            # Store change in params and gradients
            self.s_arr[k-1] = self.params - self.params_last
            self.log.debug("   S arr (change in params): %s" % self.s_arr[k-1])
            self.y_arr[k-1] = gradients - self.gradients_last
            self.log.debug("   Y arr (change in grads): %s" % self.y_arr[k-1])

            update = self.calculate_update(k, gradients, 0.0)

            # Check this is a descent direction
            eps = 1e-10
            eps_max = 1e8
            while not self.check_is_descent_direction(update, gradients) and eps <= eps_max:

                # Regularize more
                eps *= 10.0

                # Try again
                update = self.calculate_update(k, gradients, eps)

            if eps > eps_max:
                raise ValueError("Could not find a descent direction :/")

        # Update last
        self.params_last = self.params.copy()
        self.gradients_last = gradients.copy()

        # New params
        self.params += update

        # Enforce bounds as needed
        if enforce_bounds:
            assert len(bounds) == self.no_params

            for i in range(0,self.no_params):
                self.params[i] = max(bounds[i,0]+tol_bounds,self.params[i])
                self.params[i] = min(bounds[i,1]-tol_bounds,self.params[i])
            update = self.params - self.params_last

        self.log.debug("   Old params: %s" % self.params_last)
        self.log.debug("   New params: %s" % self.params)

        # Remove old ones
        idx_remove = k - self.m -1
        if idx_remove >= 0:
            del self.s_arr[idx_remove]
            del self.y_arr[idx_remove]
        
        self.log.debug("Iteration: %d [finished]" % k)

        # Monitor convergence
        if np.max(abs(update)) < tol:
            raise ZeroUpdate()

        return update

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
            traj[0] = self.params.copy()

        update = np.zeros(len(params_init))
        for k in range(0,no_steps):

            try:
                update = self.step(
                    k=k,
                    tol=tol,
                    gradient_noise_mag=gradient_noise_mag,
                    enforce_bounds=enforce_bounds,
                    bounds=bounds
                    )
            except ZeroUpdate:
                self.log.info("Converged because zero update")
                return (True, k, update, traj)
            except ZeroDescentProduct:
                self.log.info("Converged because zero descent product")
                return (True, k, update, traj)

            if store_traj:
                traj[k+1] = self.params.copy()
        
        return (False, no_steps, update, traj)