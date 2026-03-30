import numpy as np
import traceback
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import StatePreparation

try:
    print("Testing qiskit aer ...")
    simulator = AerSimulator()
    qc = QuantumCircuit(4, 4)
    c = np.ones(16)/4.0
    x = np.ones(16)/4.0
    x[0] = 0
    x = x / np.linalg.norm(x)
    
    sp_c = StatePreparation(c)
    
    # Workaround for aer segfault on inverted state preparation
    sp_x_qc = QuantumCircuit(4)
    sp_x_qc.append(StatePreparation(x), range(4))
    # Decompose heavily so it's just standard gates
    sp_x_qc = transpile(sp_x_qc, basis_gates=['cx', 'rx', 'ry', 'rz', 'h', 'u'], optimization_level=3)
    
    qc.append(sp_c, range(4))
    qc.compose(sp_x_qc.inverse(), range(4), inplace=True)
    qc.measure(range(4), range(4))
    
    print("Circuit built.")
    tc = transpile(qc, simulator)
    print("Circuit transpiled.")
    result = simulator.run(tc, shots=100).result()
    print("Simulator finished.", result.get_counts())
except Exception as e:
    print("Error:", repr(e))
    traceback.print_exc()
