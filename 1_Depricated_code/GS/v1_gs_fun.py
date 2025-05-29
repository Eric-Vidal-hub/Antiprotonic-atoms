# in case that they want to be finally implemented for any of the trust minimizers
    def jacobian(self, xx, e_num, e_spin):
        # Compute the gradient of the Hamiltonian with respect to xx
        grad = np.zeros_like(xx)
        
        # Small perturbation for numerical differentiation
        epsilon = 1e-8
        
        # Compute the gradient numerically
        for i in range(len(xx)):
            xx_plus = np.copy(xx)
            xx_minus = np.copy(xx)
            xx_plus[i] += epsilon
            xx_minus[i] -= epsilon
            
            h_plus = self.hamiltonian(xx_plus, e_num, e_spin)
            h_minus = self.hamiltonian(xx_minus, e_num, e_spin)
            
            grad[i] = (h_plus - h_minus) / (2 * epsilon)
        
        return grad
    
    def hessian(self, xx, e_num, e_spin):
        # Compute the Hessian of the Hamiltonian with respect to xx
        hess = np.zeros((len(xx), len(xx)))
        
        # Small perturbation for numerical differentiation
        epsilon = 1e-8
        
        # Compute the Hessian numerically
        for i in range(len(xx)):
            for j in range(len(xx)):
                xx_plus_plus = np.copy(xx)
                xx_plus_minus = np.copy(xx)
                xx_minus_plus = np.copy(xx)
                xx_minus_minus = np.copy(xx)
                
                xx_plus_plus[i] += epsilon
                xx_plus_plus[j] += epsilon
                
                xx_plus_minus[i] += epsilon
                xx_plus_minus[j] -= epsilon
                
                xx_minus_plus[i] -= epsilon
                xx_minus_plus[j] += epsilon
                
                xx_minus_minus[i] -= epsilon
                xx_minus_minus[j] -= epsilon
                
                h_plus_plus = self.hamiltonian(xx_plus_plus, e_num, e_spin)
                h_plus_minus = self.hamiltonian(xx_plus_minus, e_num, e_spin)
                h_minus_plus = self.hamiltonian(xx_minus_plus, e_num, e_spin)
                h_minus_minus = self.hamiltonian(xx_minus_minus, e_num, e_spin)
                
                hess[i, j] = (h_plus_plus - h_plus_minus - h_minus_plus + h_minus_minus) / (4 * epsilon**2)
        
        return hess