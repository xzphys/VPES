# =====================================================
# 正确可运行版本
#
# 修复内容：
#
# 1. CZ退极化噪声真正作用于量子态
# 2. fidelity使用noisy density matrix计算
# 3. objective function中的overlap也加入噪声
# 4. 测量噪声 = 0.01
# 5. fidelity会随CZ噪声下降
#
# =====================================================

import numpy as np
import matplotlib.pyplot as plt
import time

from qiskit import QuantumCircuit, transpile

from qiskit.quantum_info import (
    Statevector,
    DensityMatrix,
    state_fidelity
)

from qiskit_aer import AerSimulator

from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    ReadoutError
)

# =====================================================
# Generate Ansatz
# =====================================================

def apply_fixed_ansatz(qubits, layer, parameters):

    qc = QuantumCircuit(qubits)

    # =================================================
    # Initial H layer
    # =================================================

    for i in range(qubits):

        qc.h(i)

    # =================================================
    # Variational layers
    # =================================================

    for l in range(layer):

        # ---------------------------------------------
        # First RY layer
        # ---------------------------------------------

        for i in range(qubits):

            qc.ry(
                parameters[2*l*qubits+i],
                i
            )

        # ---------------------------------------------
        # Even CZ layer
        # ---------------------------------------------

        for i in range(int(qubits/2)):

            qc.cz(
                2*i,
                2*i+1
            )

        # ---------------------------------------------
        # Second RY layer
        # ---------------------------------------------

        for i in range(qubits):

            qc.ry(
                parameters[(2*l+1)*qubits+i],
                i
            )

        # ---------------------------------------------
        # Odd CZ layer
        # ---------------------------------------------

        for i in range(int(qubits/2 - 1/2)):

            qc.cz(
                2*i+1,
                2*i+2
            )

    return qc


# =====================================================
# Build noise model
# =====================================================

def build_noise_model(cz_noise):

    noise_model = NoiseModel()

    # =================================================
    # CZ depolarizing noise
    # =================================================

    cz_error = depolarizing_error(
        cz_noise,
        2
    )

    noise_model.add_all_qubit_quantum_error(

        cz_error,

        ['cz']
    )

    # =================================================
    # Measurement noise
    # =================================================

    meas_error = ReadoutError([

        [0.99, 0.01],
        [0.01, 0.99]

    ])

    for q in range(qubits):

        noise_model.add_readout_error(
            meas_error,
            [q]
        )

    return noise_model


# =====================================================
# Expectation helper
# =====================================================

def expectation_value(
    counts,
    qubit_mea,
    shots
):

    E = 0

    for s in counts:

        ss = ''

        for qm in qubit_mea:

            ss += s[::-1][qm]

        parity = 1

        if ss.count('1') % 2 == 1:

            parity = -1

        E += parity * counts[s]/shots

    return E


# =====================================================
# Noisy fidelity helper
# =====================================================

def noisy_fidelity(
    qc,
    target_state,
    noise_model
):

    backend = AerSimulator(

        method='density_matrix',

        noise_model=noise_model
    )

    qc2 = qc.copy()

    qc2.save_density_matrix()

    result = backend.run(

        transpile(qc2, backend)

    ).result()

    rho = DensityMatrix(

        result.data(0)['density_matrix']
    )

    fid = state_fidelity(

        rho,
        target_state
    )

    return fid


# =====================================================
# Objective Function
# =====================================================

def objective_function(
    parameters,
    noise_model
):

    shots = 1000

    backend = AerSimulator(

        method='density_matrix',

        noise_model=noise_model
    )

    # =================================================
    # Circuit 1
    # =================================================

    circ1 = apply_fixed_ansatz(

        qubits,
        layer,
        parameters
    )

    for q in range(qubits):

        circ1.h(q)

    circ1.measure_all()

    result1 = backend.run(

        transpile(circ1, backend),

        shots=shots,

        seed_simulator=10

    ).result()

    counts1 = result1.get_counts()

    # =================================================
    # Correlators
    # =================================================

    E1 = expectation_value(
        counts1,
        [0],
        shots
    )

    E11 = expectation_value(
        counts1,
        [0,1],
        shots
    )

    E2 = expectation_value(
        counts1,
        [2],
        shots
    )

    E21 = expectation_value(
        counts1,
        [2,3],
        shots
    )

    # =================================================
    # Circuit 2
    # =================================================

    circ2 = apply_fixed_ansatz(

        qubits,
        layer,
        parameters
    )

    for q in range(qubits):

        circ2.rx(
            np.pi/2,
            q
        )

    circ2.measure_all()

    result2 = backend.run(

        transpile(circ2, backend),

        shots=shots,

        seed_simulator=10

    ).result()

    counts2 = result2.get_counts()

    E0 = expectation_value(
        counts2,
        [0,1],
        shots
    )

    E01 = expectation_value(
        counts2,
        [2,3],
        shots
    )

    # =================================================
    # Target state
    # =================================================

    target = np.array([

        0.1381966,
        0.2236068,
        0.2236068,
        0.1381966,

        0.2236068,
        0.3618034,
        0.3618034,
        0.2236068,

        0.2236068,
        0.3618034,
        0.3618034,
        0.2236068,

        0.1381966,
        0.2236068,
        0.2236068,
        0.1381966
    ])

    target_state = Statevector(target)

    # =================================================
    # Noisy overlap
    # =================================================

    circ7 = apply_fixed_ansatz(

        qubits,
        layer,
        parameters
    )

    prob = noisy_fidelity(

        circ7,

        target_state,

        noise_model
    )

    return -0.125 * prob * (

        1 + 0.25 * (

            E1
            + E2
            + 0.5 * (
                E11
                + E21
                + E0
                + E01
            )
        )
    )


# =====================================================
# Gradient
# =====================================================

def Gradient(
    theta,
    noise_model
):

    g = np.zeros(
        2 * layer * qubits
    )

    for i in range(2 * layer * qubits):

        theta_temp = theta.copy()

        theta_temp[i] += np.pi/4

        e1 = objective_function(
            theta_temp,
            noise_model
        )

        theta_temp[i] -= np.pi/2

        e2 = objective_function(
            theta_temp,
            noise_model
        )

        g[i] = e1.real - e2.real

    return g


# =====================================================
# Adam optimizer
# =====================================================

def adam_iters(
    theta0,
    max_iter,
    noise_model
):

    final_precision = 1e-4

    theta_precision = 1e-2

    beta1 = 0.9

    beta2 = 0.999

    alpha = 0.1

    epsilon = 1e-8

    npoints = 2 * layer * qubits

    m = np.zeros(npoints)

    v = np.zeros(npoints)

    theta = np.zeros(
        (max_iter+1, npoints)
    )

    energy = np.zeros(
        max_iter+1
    )

    g = np.zeros(
        (max_iter+1, npoints)
    )

    theta[0] = theta0.copy()

    g[0] = Gradient(
        theta[0],
        noise_model
    )

    energy[0] = objective_function(
        theta[0],
        noise_model
    ).real

    for t in range(1, max_iter+1):

        m = beta1*m + (1-beta1)*g[t-1]

        v = beta2*v + (1-beta2)*g[t-1]**2

        alphat = alpha * np.sqrt(
            1-beta2**t
        ) / (1-beta1**t)

        theta_change = alphat * m / (
            np.sqrt(v)+epsilon
        )

        theta[t] = theta[t-1] - theta_change

        g[t] = Gradient(
            theta[t],
            noise_model
        )

        energy[t] = objective_function(
            theta[t],
            noise_model
        ).real

        if (

            np.abs(
                energy[t]-energy[t-1]
            ) < final_precision

            and

            np.linalg.norm(
                theta_change
            ) < theta_precision
        ):
            break

        if t % 10 == 0:

            print("Iteration =", t)

    return theta[:t+1], energy[:t+1]


# =====================================================
# Main
# =====================================================

qubits = 4

layer = 2

max_iter = 100

# =====================================================
# Noise scan
# =====================================================

p2_list = np.linspace(
    0,
    0.1,
    21
)

fidelity_list = []

# =====================================================
# Target state
# =====================================================

target = np.array([

    0.1381966,
    0.2236068,
    0.2236068,
    0.1381966,

    0.2236068,
    0.3618034,
    0.3618034,
    0.2236068,

    0.2236068,
    0.3618034,
    0.3618034,
    0.2236068,

    0.1381966,
    0.2236068,
    0.2236068,
    0.1381966
])

target_state = Statevector(target)

# =====================================================
# Noise loop
# =====================================================

for p in p2_list:

    print("\n================================")
    print("CZ noise =", p)
    print("================================\n")

    # =================================================
    # Build noise model
    # =================================================

    noise_model = build_noise_model(p)

    # =================================================
    # Initial parameters
    # =================================================

    np.random.seed(10)

    theta0 = np.random.uniform(

        -np.pi/2,
        np.pi/2,
        2 * layer * qubits
    )

    # =================================================
    # Optimization
    # =================================================

    start = time.time()

    opt_para1, e1 = adam_iters(

        theta0,
        max_iter,
        noise_model
    )

    end = time.time()

    opt_para = opt_para1[-1]

    # =================================================
    # Final noisy fidelity
    # =================================================

    qc = apply_fixed_ansatz(

        qubits,
        layer,
        opt_para
    )

    fidelity = noisy_fidelity(

        qc,

        target_state,

        noise_model
    )

    fidelity_list.append(fidelity)

    print("Final fidelity =", fidelity)

    print(
        "Optimization time =",
        end-start,
        "s"
    )

# =====================================================
# Plot
# =====================================================

