import numpy as np
from typing import Optional


class LinearSolver:
    """
    Create a LinearSolver object with different methods to solve the regularized least squares problem.
    """

    def __init__(self, A: np.ndarray, y: np.ndarray) -> None:
        if A.ndim != 2:
            raise ValueError("Initialization failed because the A matrix is not a 2D numpy array.")
        if y.ndim != 1:
            y = y.reshape(-1)
            print("Warning: vector is not a 1D numpy array. Reshaped to 1D numpy array automatically.")
        if A.shape[0]!=y.shape[0]:
            raise ValueError("Mismatched number of rows. Initialization failed.")
        self.A = A
        self.y = y
        return None

    def rls(self, reg: Optional[float] = 1.0) -> np.ndarray:
        """
        Compute solution to the regularized least squares problem.
        
        :param 
            - self: A LinearSolver object with an A attribute (a numpy array of shape (m,d)) and a y attribute (a numpy array of shape(m,))
            - reg: An optional numeric value that serves as a regularization parameter for solving the RLS problem.
        :returns: 
            - The solution to the unregularized least squares solution (of shape (d,)) if reg is None or zero
            - The solution to the regularized least squares solution (of shape (d,)) if reg is positive
        :raises ValueError: if reg is negative
        
        """
        if reg is None or reg==0:
            x = np.linalg.solve(self.A.T@self.A, self.A.T@self.y)
        elif reg < 0:
            raise ValueError('Regularization parameter should be nonnegative or None.')
        elif reg>0:
            x = np.linalg.solve(self.A.T@self.A+reg*np.eye(self.A.shape[1]), self.A.T@self.y)
        return x

    def sgd(self,
            reg: float = 1.0,
            max_iter: int = 100,
            batch_size: int = 2,
            step_size: float = 1e-2) -> np.ndarray:
        """
        Perform mini-batch Stochastic Gradient Descent (SGD) to solve the RLS problem.

        :param
            - self: A LinearSolver object with an A attribute (a numpy array of shape (m,d)) and a y attribute (a numpy array of shape(m,))
            - reg: A numeric value that serves as a regularization parameter for solving the RLS problem.
            - max_iter: A positive integer that determines how many iterations of SGD we want to perform.
            - batch_size: A positive integer that determines the number of random rows of A and y used to form A_b and y_b respectively.
            - step_size: A numeric value that determines the learning rate of SGD.
        : returns:
            - The coefficient vector x as a numpy array of shape (d,)
        :raises ValueError: if reg is negative
        
        """
        if reg is None or reg==0 or reg>0:
            x = np.zeros(self.A.shape[1])
            for i in range(max_iter):
                b = np.random.choice(self.A.shape[0], size=batch_size, replace=False) 
                A_b = self.A[b,:]
                y_b = self.y[b]
                x = x - step_size*(2*A_b.T@(A_b@x-y_b)+2*reg*x)
        elif reg<0:
            raise ValueError('Regularization parameter should be nonnegative or None.')
        return x
