An New Variational Quantum Algorithm for Solving the Poisson equation

A Qiskit-implemented variational quantum algorithm designed to solve the Poisson equation. We first adopt the finite difference method to transform the Poisson equation into a linear system. Then, adopt the mixed expansion method of the coefficient matrix in the Pauli basis and computational basis, and use the joint measurement of the Pauli basis and Bell basis to obtain the objective function value of the variational quantum algorithm. The hybrid measurement scheme can reduce the usage of quantum resources. This implementation leverages the hardware efficient ansatz, quantum state measurements, and the ADAM optimization algorithm to find optimal quantum circuit parameters. 

Key features:
1. The hardware efficient ansatz with RY rotations, and CZ entangling gates
2. ADAM gradient descent optimizer for efficient parameter updates
3. Expectation value measurements of multiple qubit combinations to obtain the value of objective function
4. The file VPES.py does not introduce noise, whereas VPES_noise.py incorporates noise.

Code Structure
1. The hardware efficient ansatz to generate an parameterized quantum state
Initialization: Apply Hadamard gates to all qubits to create a superposition state.
The hardware efficient ansatz with an alternating layered ansatz consisting of RY gates and controlled Z gate 
2. Objective Function
Since the coefficient matrix is expanded in the Pauli basis and the computational basis, joint measurement of the Pauli basis and the Bell basis is adopted according to its form to obtain the value of the objective function. Here, we  provides the qiskit code of variational quantum algorithm to solve a 16 *16  linear system Ax=b using 4 qubits.
3. Gradient Calculation: Computes gradients using the Parameter Shift method
4. ADAM Optimization: Implements the ADAM gradient descent algorithm for parameter optimization

Main Execution
(1) Parameter Initialization
Fixed parameters: qubits=4 , layer=2 (ansatz layers), max_iter=100 (max optimization iterations)
Initial parameter initialization: Random values in [-pi/2, pi/2]
(2) Run Optimization
Run adam_iters(theta0, max_iter) to obtain optimized parameters and the value of objective function
(3) Extract Results
Constructs the ansatz with optimized parameters and extracts the final quantum state
Computes the absolute overlap (f) between the final state and the target vector
(4) Final outputs: 
Otput the final state overlap (f), minimized energy (e), optimal parameters (opt_para), and the final quantum state vector (b).

Notes
Simulator Dependence: Uses Qiskit's qasm_simulator
Reproducibility: Fixed seed_simulator and np.random.seed(10) ensure consistent results across runs.
