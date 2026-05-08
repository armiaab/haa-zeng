from utils import *
import pandas as pd

molecule = "H2"
basis = "sto-3g"

def get_geometry(r):
    return f"H 0 0 0; H 0 0 {r}"


r_list = np.linspace(0.5, 3.0, 10)
energies = []

layer = 1
ancilla = 1
maxiter = 100

warm_start_params = {
    'haa_can_cc':          None,
    'haa_can_ring':        None,
    'haa_can_alternate':   None,
    'haa_can_full':        None,
    'haa_can_skip':        None,
    'haa_givens_cc':       None,
    'haa_givens_ring':     None,
    'haa_givens_skip':     None,
    'haa_givens_full':     None,
    'haa_fsim_cc':         None,
    'haa_fsim_full':       None,
    'haa_su4_cc':          None,
    'haa_mixed':           None,
    'hea':                 None,
    'qrqnn':               None,
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

    circuits = {
        'haa_can_cc':        build_haa(n_sys, ancilla, layer, hf_circuit, internal="can",    coupling="cc"),
        'haa_can_ring':      build_haa(n_sys, ancilla, layer, hf_circuit, internal="can",    coupling="ring"),
        'haa_can_alternate': build_haa(n_sys, ancilla, layer, hf_circuit, internal="can",    coupling="alternate"),
        'haa_can_full':      build_haa(n_sys, ancilla, layer, hf_circuit, internal="can",    coupling="full"),
        'haa_can_skip':      build_haa(n_sys, ancilla, layer, hf_circuit, internal="can",    coupling="skip"),
        'haa_givens_cc':     build_haa(n_sys, ancilla, layer, hf_circuit, internal="givens", coupling="cc"),
        'haa_givens_ring':   build_haa(n_sys, ancilla, layer, hf_circuit, internal="givens", coupling="ring"),
        'haa_givens_skip':   build_haa(n_sys, ancilla, layer, hf_circuit, internal="givens", coupling="skip"),
        'haa_givens_full':   build_haa(n_sys, ancilla, layer, hf_circuit, internal="givens", coupling="full"),
        'haa_fsim_cc':       build_haa(n_sys, ancilla, layer, hf_circuit, internal="fsim",   coupling="cc"),
        'haa_fsim_full':     build_haa(n_sys, ancilla, layer, hf_circuit, internal="fsim",   coupling="full"),
        'haa_su4_cc':        build_haa(n_sys, ancilla, layer, hf_circuit, internal="su4",    coupling="cc"),
        'haa_mixed':         build_haa(n_sys, ancilla, 3,     hf_circuit,
                                       internal=["givens", "fsim",  "su4"],
                                       coupling=["cc",     "ring",  "full"]),
        'hea':               build_hea(n_sys, layer, hf_circuit),
        'qrqnn':             build_qrqnn(n_sys, ancilla, layer, hf_circuit),
    }

    results = {}
    for key, circuit in circuits.items():
        n_anc = 0 if key == 'hea' else ancilla
        energy, params = run_vqe(
            circuit, qop,
            lambda: ADAM(maxiter=maxiter),
            n_anc=n_anc,
            enuc=enuc,
            e_off=e_off,
            warm_start=warm_start_params[key]
        )
        results[key] = energy
        warm_start_params[key] = params

    results['hf']  = E_HF
    results['fci'] = E_FCI
    energies.append(results)
    print(f"r={r:.2f} | " + " | ".join(f"{k}={v:.6f}" for k, v in results.items()))


for r in r_list:
    run_experiment(r)


df = pd.DataFrame({
    "r":                    r_list,
    "energy_haa_can_cc":    [e['haa_can_cc']        for e in energies],
    "energy_haa_can_ring":  [e['haa_can_ring']       for e in energies],
    "energy_haa_can_alt":   [e['haa_can_alternate']  for e in energies],
    "energy_haa_can_full":  [e['haa_can_full']       for e in energies],
    "energy_haa_can_skip":  [e['haa_can_skip']       for e in energies],
    "energy_haa_givens_cc": [e['haa_givens_cc']      for e in energies],
    "energy_haa_givens_ring":[e['haa_givens_ring']   for e in energies],
    "energy_haa_givens_skip":[e['haa_givens_skip']   for e in energies],
    "energy_haa_givens_full":[e['haa_givens_full']   for e in energies],
    "energy_haa_fsim_cc":   [e['haa_fsim_cc']        for e in energies],
    "energy_haa_fsim_full": [e['haa_fsim_full']      for e in energies],
    "energy_haa_su4_cc":    [e['haa_su4_cc']         for e in energies],
    "energy_haa_mixed":     [e['haa_mixed']           for e in energies],
    "energy_hea":           [e['hea']                for e in energies],
    "energy_qrqnn":         [e['qrqnn']              for e in energies],
    "energy_hf":            [e['hf']                 for e in energies],
    "energy_fci":           [e['fci']                for e in energies],
})

df.to_csv(f"{molecule}_{basis}_full.csv", index=False)
print(f"\n{molecule}_{basis}_full.csv")