from utils import *
import pandas as pd

molecule = "H2"
basis = "sto-3g"

def get_geometry(r):
    return f"H 0 0 0; H 0 0 {r}"


r_list = np.linspace(0.5, 3.0, 10)
energies = []
expressibilities = []

layer = 1
ancilla = 1
maxiter = 2000

# Storage for warm start parameters from previous geometry
warm_start_params = {
    'haa_can': None,
    'haa_u3cx': None,
    'hea': None,
    'qrqnn': None
}

def run_experiment(r):
    geometry = get_geometry(r)
    mol = gto.M(atom=geometry, basis=basis, unit="Angstrom")
    mf = scf.RHF(mol).run(verbose=0)
    E_HF, E_FCI = mf.e_tot, fci.FCI(mf).kernel()[0]

    problem = PySCFDriver(atom=geometry, basis=basis, unit=DistanceUnit.ANGSTROM).run()
    mapper = JordanWignerMapper()

    qop = mapper.map(problem.hamiltonian.second_q_op())
    enuc = problem.hamiltonian.nuclear_repulsion_energy
    hf_circuit = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
    e_off = problem.reference_energy - float(np.real(Statevector(hf_circuit).expectation_value(qop))) - enuc
    n_sys = qop.num_qubits

    haa_circuit_can = build_haa(n_sys, ancilla, layer, hf_circuit)
    haa_circuit_u3cx = build_haa(n_sys, ancilla, layer, hf_circuit, internal="u3cx", coupling="ac")
    hea_circuit = build_hea(n_sys, layer, hf_circuit)
    qrqnn_circuit = build_qrqnn(n_sys, ancilla, layer, hf_circuit)
    
    energy_haa_can, params_haa_can = run_vqe(haa_circuit_can, qop, lambda: ADAM(maxiter=maxiter), ancilla, enuc=enuc, e_off=e_off, warm_start=warm_start_params['haa_can'])
    energy_haa_u3cx, params_haa_u3cx = run_vqe(haa_circuit_u3cx, qop, lambda: ADAM(maxiter=maxiter), ancilla, enuc=enuc, e_off=e_off, warm_start=warm_start_params['haa_u3cx'])
    energy_hea, params_hea = run_vqe(hea_circuit, qop, lambda: ADAM(maxiter=maxiter), enuc=enuc, e_off=e_off, warm_start=warm_start_params['hea'])
    energy_qrqnn, params_qrqnn = run_vqe(qrqnn_circuit, qop, lambda: ADAM(maxiter=maxiter), ancilla, enuc=enuc, e_off=e_off, warm_start=warm_start_params['qrqnn'])
    
    warm_start_params['haa_can'] = params_haa_can
    warm_start_params['haa_u3cx'] = params_haa_u3cx
    warm_start_params['hea'] = params_hea
    warm_start_params['qrqnn'] = params_qrqnn
    
    energies.append((energy_haa_can, energy_haa_u3cx, energy_hea, energy_qrqnn, E_HF, E_FCI))
    print(f"r={r:.2f}: HAA-CAN={energy_haa_can:.4f}, HAA-U3CX={energy_haa_u3cx:.4f}, HEA={energy_hea:.4f}, QRQNN={energy_qrqnn:.4f}")


for r in r_list:
    run_experiment(r)

df = pd.DataFrame({
    "r": r_list,
    "energy_haa_can": [e[0] for e in energies],
    "energy_haa_u3cx": [e[1] for e in energies],
    "energy_hea": [e[2] for e in energies],
    "energy_qrqnn": [e[3] for e in energies],
    "energy_hf": [e[4] for e in energies],
    "energy_fci": [e[5] for e in energies],
})

df.to_csv(f"{molecule}_{basis}.csv", index=False)
