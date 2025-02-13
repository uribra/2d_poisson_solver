# Created by Uriel Braham

# Import necessary modules
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class FD_POISSON:

    '''
    Class to solve the Poisson equation with Dirichlet boundary data

            - (u_xx + u_yy) + c*u  = f        in (0,L_x) x (0,L_y)
                        u(x,y)      = g       on space boundary 
                     
    with source source f rectangle (0,L_x) x (0,L_y) using central in space difference quotients for space discretization. 

    
    *Args:
    - L_x (float):                              Length of rectangle in x-direction
    - L_y (float):                              Length of rectangle in y-direction
    - h (float):                                Space discretization parameter
    - f (function f = f(x,y) or array):         Source
    - g (function g = g(x,y) or array):         Boundary data
    

    * Return: 
    - plot_3D_solution() returns 3D plot of the finite difference solution
    - plot_contour() returns contour plot of the finite difference solution

    
    * Utilities:
    - asvector() reshapes a matrix into a vector using lexigraphic ordering
    - asmatrix() returns the matrix version of a vector 
    - MESH() returns 2D mesh on [0,L_x] x [0,L_y]
    - INTERIOR_MESH() returns 2D mesh on [0,L_x] x [0,L_y]
    - GENERATE_STIFFNESS_MATRIX() returns the stiffness matrix for the discrete Laplacian
    - DISCRETIZE_F(), DISCRETIZE_G(), DISCRETIZE_C() return the grid/boundary grid functions for f, g and c
    - ASSEMBLE_RHS_VECTOR() assembles the new rhs with the boundary data
    - GENERATE_C_COEFFICIENT() generates the coefficient matrix associated to the coefficient function c
    - SOLVE_DISCRETE_SYSTEM() returns grid function on the discretized space domain [0,L_x] x [0,L_y]

    '''
  
    def __init__(self, L_x, L_y, h, c, f, g):

        self.L_x = L_x
        self.L_y = L_y
        self.h = h
        self.f = f 
        self.c = c
        self.g = g 

        # Controlstructres for args* 
        if L_x < 0 or L_y<0:
            raise ValueError("Domain parameters incorrect. Expected L_x,L_y>0.")
        if h <= 0:
            raise ValueError("Space discretization parameters non-positive. Expected h>0.")
        if not callable(f) or isinstance(f, (list, tuple, np.ndarray)):
            raise TypeError("Source has to be a function f(x,y) or a grid function on the interior grid points as list or 2D array") 
        if not callable(g) or isinstance(g, (list, tuple, np.ndarray)):
            raise TypeError("Boundary data has to be a function g(x,y) or a grid function on the boundary grid points as list or 2D array") 
        if not callable(c) or isinstance(c, (list, tuple, np.ndarray)):
            raise TypeError("Coefficient function c has to be a function c(x,y) or a grid function on the interior grid points as list or 2D array") 
        
        self.n_x = int(L_x/h)       # Number of grid points in x-direction 
        self.n_y = int(L_y/h)       # Number of grid points in x-direction 


    def asvector(self, M):
        m = M.shape[0] # number of rows
        n = M.shape[1] # number of colums
        vec_M = np.zeros(n*m) 
        for i in range(0,n):
            vec_M[i*m:(i+1)*m] = M[:,i]
        return vec_M


    def asmatrix(self, V, m, n):
        M = np.zeros((m,n)) 
        for i in range(0,n):
            M[0:m,i] = V[i*m:(i+1)*m]
        return M


    def MESH(self):
        x = np.linspace(0, self.L_x, self.n_x+1)
        y = np.linspace(0, self.L_y, self.n_y +1)
        X,Y = np.meshgrid(x,y)
        return X,Y
    

    def INTERIOR_MESH(self):
        x = np.linspace(0, self.L_x, self.n_x+1)
        y = np.linspace(0, self.L_y, self.n_y +1)
        X_INT,Y_INT = np.meshgrid(x[1:-1],y[1:-1])          # Interior mesh points

        return X_INT,Y_INT
    

    def DISCRETIZE_F(self):
        f = np.vectorize(self.f)
        X_int,Y_int = self.INTERIOR_MESH()              # Meshgrid for interior points
        F = f(X_int,Y_int)                         # Evaluate f on the interior grid point
        return F
    

    def DISCRETIZE_G(self):
        X,Y = self.MESH()                           
        g = self.g
        # Define a new g: g_new = g on the boundary and otherwise g=0 on the interior nodes
        def g_new(x,y):
            if y==0:
                return g(x,y)
            elif y==self.L_y:
                return g(x,y)
            elif x==0:
                return g(x,y)
            elif x==self.L_x:
                return g(x,y)
            else: 
                return 0    
        g_new = np.vectorize(g_new)
        G = g_new(X,Y)                                   # Evaluate g on the interior grid point
        return G

    def DISCRETIZE_C(self): 
        c = np.vectorize(self.c)
        X_int,Y_int = self.INTERIOR_MESH()
        C = c(X_int,Y_int)                               # Evaluate f on the interior grid point
        return C
    
    '''
    def DISCRETIZE_B(self): 
        n_x = self.n_x
        n_y = self.n_y
        L_x =self.L_x
        L_y =self.L_y
        b = self.b
        
        x = np.linspace(0,L_x,n_x+1)
        y = np.linspace(0,L_y, n_y +1)
        X_int,Y_int = np.meshgrid(x[1:-1],y[1:-1])       # Meshgrid for interior points
        B = b(X_int,Y_int)                               # Evaluate f on the interior grid point
        return B
    '''

    def GENERATE_STIFFNESS_MATRIX(self):

        n_x = self.n_x
        n_y = self.n_y
        h  = self.h

        # Initialize the 5-point stencil matrix for the discrete Laplacian
        B_h = np.zeros((n_y-1, n_y-1))
        B_h[-1,-1] = 4
        for i in range(0,n_x-2):
            B_h[i,i] = 4
            B_h[i+1,i] = -1
            B_h[i,i+1] = -1

        # Generate the stiffness matrix 
        stiffness = np.zeros(((n_y-1)*(n_x-1), (n_y-1)*(n_x-1)))

        # Generate the Blocks B_h on the blockdiagonal of the stiffness matrix 
        for i in range(0,n_x-1):
            stiffness[i*(n_x-1):(i+1)*(n_x-1), i*(n_x-1):(i+1)*(n_x-1)] = B_h

        # Generate a negative identity matrix negI
        negI = np.zeros((n_y-1,n_y-1))
        for i in range(0,n_y-1):
            negI[i,i] = -1

        # Generate the negative identity blocks on the lower/upper blockdiagonal 
        for i in range(0,n_x-2):
            stiffness[(i+1)*(n_y-1):(i+2)*(n_y-1), i*(n_y-1):(i+1)*(n_y-1)] = negI  # lower
            stiffness[i*(n_y-1):(i+1)*(n_y-1), (i+1)*(n_y-1):(i+2)*(n_y-1)] = negI  # upper

        return (1/(h**2))*stiffness
    

    def GENERATE_C_COEFFICIENT(self):
        C = self.DISCRETIZE_C()
        vec_C = self.asvector(C)                    # Vectorize the initial condition 
        C_MATRIX = np.zeros(((self.n_y-1)*(self.n_x-1), (self.n_y-1)*(self.n_x-1)))
        for i in range(0,(self.n_y-1)**2):     
            C_MATRIX[i,i] = vec_C[i]
        return C_MATRIX
    

    def ASSEMBLE_RHS_VECTOR(self): 

        n_x = self.n_x
        n_y = self.n_y
        h = self.h   

        F = self.DISCRETIZE_F()
        G = self.DISCRETIZE_G()

        r = np.zeros((n_y-1)*(n_x-1))
        
        temp1 = np.zeros(n_y-1)
        temp1[0] = G[0,1]
        temp1[-1] = G[n_y,1]
        r[0:n_y-1] = F[:,0] + (G[1:-1, 0] + temp1)/h**2

        temp2 = np.zeros(n_y-1)
        temp2[0] = G[0,-1]
        temp2[-1] = G[-1,-1] 
        r[(n_y-1)*(n_x-1) - (n_y-1):(n_y-1)*(n_x-1)] = F[:, n_y-2] + (G[1:-1,n_x] + temp2)/h**2

        for j in range(1,n_x-2):
            temp = np.zeros(n_y-1)
            temp[0] = G[0,j]
            temp[-1] = G[-1,j]
            r[j*(n_y-1):(j+1)*(n_y-1)] = F[:,j] + temp/h**2

        return r


    def SOLVE_DISCRETE_SYSTEM(self):

        # Discretize the right hand side f on the interior grid points and store as vector
        #F = self.DISCRETIZE_F()
        G = self.DISCRETIZE_G()

        r = self.ASSEMBLE_RHS_VECTOR()

        C_COEFFICIENT = self.GENERATE_C_COEFFICIENT()
    
        # Generate stiffness matrix for the discrete Laplacian
        A_h = self.GENERATE_STIFFNESS_MATRIX()
        # Combine all matricies for the finite difference approximations of derivatives
        B = A_h + C_COEFFICIENT

        B_sparse = sp.csr_matrix(B)                           # Store as csr_compressed matrix 
        U = spsolve(B_sparse, r)                
        U_h = self.asmatrix(U, (self.n_y-1), (self.n_x-1))    # Assemble the solution vector into 2D-array/grid function

        # Create 2D-array with solution including zero boundary conditions
        solution = np.zeros((self.n_x+1,self.n_y+1))

        # Add solution for interior grid points
        solution[1:-1,1:-1] = U_h  
        # Add boundary conditions to the solution 
        solution[:,0] = G[:,0]
        solution[:,self.n_x] = G[:,self.n_x]
        solution[0,1:self.n_x] = G[0,1:self.n_x]
        solution[self.n_y,1:self.n_x] = G[self.n_y,1:self.n_x]

        return solution


    def plot_3D_solution(self):

        X,Y = self.MESH()
        U = self.SOLVE_DISCRETE_SYSTEM()

        # Create a figure and manually add with 3D projection
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d',)
        ax.plot_surface(X, Y, U, cmap='viridis')
        ax.set_title('Finite Difference Solution')
        z_min, z_max = np.min(U), np.max(U)  # Get the min and max values across all time steps
        ax.set_zlim(z_min, z_max)  # Set the Z-axis limits based on your data range
        ax.set_xlabel('x-values')
        ax.set_ylabel('y-values')
        ax.set_zlabel('u(x,y)', fontsize=12,y=1.08)
        plt.tight_layout()
        plt.show()


    def plot_3D_boundary_conditions(self):

        G = self.DISCRETIZE_G()
        X,Y = self.MESH()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('Boundary Condition')
        ax.set_xlabel('x-values')
        ax.set_ylabel('y-values')
        ax.plot3D(X[0, :], Y[0, :], G[0, :], color='blue')
        # Bottom boundary (y = min)
        ax.plot3D(X[-1, :], Y[-1, :], G[-1, :], color='blue')
        # Left boundary (x = min)
        ax.plot3D(X[:, 0], Y[:, 0], G[:, 0], color='blue')
        # Right boundary (x = max)
        ax.plot3D(X[:, -1], Y[:, -1], G[:, -1], color='blue')
        # Show plot
        ax.set_zlabel('g(x,y)', fontsize=12,y=1.08)
        plt.tight_layout()
        plt.show()