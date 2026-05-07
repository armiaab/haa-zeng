import numpy as np
from scipy.sparse import issparse
from scipy.special import rel_entr
from scipy.stats import entropy

from pyscf import gto, scf, fci
from qiskit.quantum_info import SparsePauliOp, Statevector, state_fidelity
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import ADAM, COBYLA, SPSA
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.circuit.library import HartreeFock


def build_haa(n_sys, n_anc, n_layers, hf_circuit=None, internal="can", coupling="cc"):
    n_qubits = n_sys + n_anc
    if coupling == "cc":
        if n_anc == 0:
            raise ValueError("cross coupling (cc) needs at least one ancilla qubit")
        pairs = [(s, n_sys + a) for a in range(n_anc) for s in range(n_sys)]
    elif coupling == "ac":
        pairs = [(i, i + 1) for i in range(n_qubits - 1)]
    else:
        raise ValueError("coupling must be 'ac' or 'cc'")

    if internal == "can":
        n_params = 3 * n_sys * (n_layers + 1) + 3 * len(pairs) * n_layers
    elif internal == "u3cx":
        n_params = 3 * n_sys * (n_layers + 1) + 6 * len(pairs) * n_layers
    else:
        raise ValueError("internal must be 'can' or 'u3cx'")

    p = ParameterVector("θ", n_params)
    qc = QuantumCircuit(n_qubits)

    if hf_circuit is not None:
        qc.compose(hf_circuit, qubits=range(n_sys), inplace=True)

    i = 0

    for q in range(n_sys):
        qc.u(p[i], p[i+1], p[i+2], q)
        i += 3

    for _ in range(n_layers):
        for q0, q1 in pairs:
            if internal == "can":
                qc.rxx(p[i], q0, q1)
                qc.ryy(p[i+1], q0, q1)
                qc.rzz(p[i+2], q0, q1)
                i += 3

            elif internal == "u3cx":
                qc.u(p[i], p[i+1], p[i+2], q0)
                qc.u(p[i+3], p[i+4], p[i+5], q1)
                qc.cx(q0, q1)
                i += 6

        # next system layer
        for q in range(n_sys):
            qc.u(p[i], p[i+1], p[i+2], q)
            i += 3

    return qc

def build_hea(n_qubits, n_layers, hf_circuit=None):
    pairs = [(i, i + 1) for i in range(n_qubits - 1)]
    n_params = 3 * n_qubits * (n_layers + 1) + 6 * len(pairs) * n_layers
    p = ParameterVector("θ", n_params)

    qc = QuantumCircuit(n_qubits, name="HEA")

    if hf_circuit is not None:
        qc.compose(hf_circuit, qubits=range(n_qubits), inplace=True)

    i = 0

    for q in range(n_qubits):
        qc.u(p[i], p[i+1], p[i+2], q)
        i += 3

    for _ in range(n_layers):
        for q0, q1 in pairs:
            qc.u(p[i], p[i+1], p[i+2], q0)
            qc.u(p[i+3], p[i+4], p[i+5], q1)
            qc.cx(q0, q1)
            i += 6

        for q in range(n_qubits):
            qc.u(p[i], p[i+1], p[i+2], q)
            i += 3

    return qc


def build_qrqnn(
    n_sys: int,
    n_anc: int,
    n_layers: int,
    hf_circuit: QuantumCircuit | None = None
) -> QuantumCircuit:
    total_qubits = n_sys + n_anc

    n_params = 3 * total_qubits * (n_layers + 1) + 3 * n_sys * n_anc * n_layers
    params = ParameterVector("t", n_params)

    qc = QuantumCircuit(total_qubits, name="QRQNN")

    if hf_circuit is not None:
        qc.compose(hf_circuit, qubits=range(n_sys), inplace=True)

    idx = 0

    for q in range(total_qubits):
        qc.u(params[idx], params[idx + 1], params[idx + 2], q)
        idx += 3

    for _ in range(n_layers):
        if n_sys > 1:
            for q in range(n_sys - 1):
                qc.cx(q, q + 1)
            qc.cx(n_sys - 1, 0)

        for anc in range(n_anc):
            anc_q = n_sys + anc
            for sys_q in range(n_sys):
                qc.rxx(params[idx], sys_q, anc_q)
                qc.ryy(params[idx + 1], sys_q, anc_q)
                qc.rzz(params[idx + 2], sys_q, anc_q)
                idx += 3

        for q in range(total_qubits):
            qc.u(params[idx], params[idx + 1], params[idx + 2], q)
            idx += 3

    return qc



def expressibility(circuit, n_samples=500, n_bins=50, eps=1e-12, seed=None):
    n_params = circuit.num_parameters
    n_qubits = circuit.num_qubits
    N = 2 ** n_qubits
    rng = np.random.default_rng(seed)

    fidelities = []

    for _ in range(n_samples):
        theta1 = rng.uniform(0, 2 * np.pi, n_params)
        theta2 = rng.uniform(0, 2 * np.pi, n_params)

        psi1 = Statevector(circuit.assign_parameters(theta1))
        psi2 = Statevector(circuit.assign_parameters(theta2))

        fidelities.append(state_fidelity(psi1, psi2))

    fidelities = np.array(fidelities)
    fidelities = np.clip(fidelities, 0, 1)  # Ensure fidelities are in [0,1]

    hist, edges = np.histogram(fidelities, bins=n_bins, range=(0, 1))
    bin_centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]
    
    p_ansatz = hist / (n_samples * width)

    p_haar = (N - 1) * (1 - bin_centers) ** (N - 2) * width
    p_haar = p_haar / np.sum(p_haar)  # Ensure normalization

    p_ansatz = np.clip(p_ansatz, eps, None)
    p_haar = np.clip(p_haar, eps, None)
    
    p_ansatz = p_ansatz / np.sum(p_ansatz)
    p_haar = p_haar / np.sum(p_haar)

    return float(entropy(p_ansatz, p_haar))


def bp_analysis(circuit, qop, n_anc, n_samples=20, seed=None):
    rng = np.random.default_rng(seed)
    
    qop_ext = SparsePauliOp.from_list([("I"*n_anc, 1.0)]).tensor(qop) if n_anc else qop
    H = qop_ext.to_matrix()
    H = H.toarray() if issparse(H) else H
    
    all_grads = []
    n_params = circuit.num_parameters
    n_check = min(20, n_params)
    
    for _ in range(n_samples):
        params = rng.uniform(0, 2*np.pi, n_params)
        
        for i in range(n_check):
            pp, pm = params.copy(), params.copy()
            pp[i] += np.pi / 2
            pm[i] -= np.pi / 2
            
            try:
                psip = Statevector(circuit.assign_parameters(pp)).data
                psim = Statevector(circuit.assign_parameters(pm)).data
                
                # Compute gradient via parameter shift rule
                exp_p = np.real(np.vdot(psip, H @ psip))
                exp_m = np.real(np.vdot(psim, H @ psim))
                grad = (exp_p - exp_m) / 2
                
                all_grads.append(abs(grad))
            except Exception:
                # Skip if numerical issues
                continue
    
    return float(np.mean(all_grads)) if all_grads else 0.0


def run_vqe(circuit, qop, optimizer_cls, n_anc=0,
            n_restart=5, enuc=0.0, e_off=0.0, warm_start=None):
    if n_anc:
        qop_ext = qop.tensor(
            SparsePauliOp.from_list([("I"*n_anc, 1.0)])
        )
    else:
        qop_ext = qop

    qop_shift = qop_ext + SparsePauliOp.from_list([
        ("I"*qop_ext.num_qubits, enuc + e_off)
    ])

    best_energy = np.inf
    best_params = None
    num_params = circuit.num_parameters

    if warm_start is not None:
        initial_points = [warm_start] + [
            np.random.uniform(0, 2*np.pi, num_params)
            for _ in range(n_restart - 1)
        ]
    else:
        initial_points = [
            np.random.uniform(0, 2*np.pi, num_params)
            for _ in range(n_restart)
        ]

    for x0 in initial_points:
        optimizer = optimizer_cls()
        vqe = VQE(
            StatevectorEstimator(),
            circuit,
            optimizer,
            initial_point=x0
        )

        result = vqe.compute_minimum_eigenvalue(qop_shift)
        energy = result.eigenvalue.real

        if energy < best_energy:
            best_energy = energy
            best_params = result.optimal_point

    return best_energy, best_params
