import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import COBYLA
from math import pi
from qiskit.circuit.library import StatePreparation

def pad_and_normalize(x, m_qubits):
    N = len(x)
    padded = np.zeros(2**m_qubits)
    padded[:N] = x
    norm = np.linalg.norm(padded)
    if norm == 0:
        padded[0] = 1.0
    else:
        padded = padded / norm
    return padded

def encode_vector(qc, v_normalized, qubits, inverse=False):
    sp = StatePreparation(v_normalized)
    if inverse:
        sp_qc = QuantumCircuit(len(qubits))
        sp_qc.append(sp, range(len(qubits)))
        sp_qc = transpile(sp_qc, basis_gates=['cx', 'rx', 'ry', 'rz', 'h', 'u'], optimization_level=3)
        qc.compose(sp_qc.inverse(), qubits, inplace=True)
    else:
        qc.append(sp, qubits)

def get_uu_dag_circuit(centroid, x, m_qubits):
    qc = QuantumCircuit(m_qubits, m_qubits)
    c_pad = pad_and_normalize(centroid, m_qubits)
    x_pad = pad_and_normalize(x, m_qubits)
    
    encode_vector(qc, c_pad, range(m_qubits))
    encode_vector(qc, x_pad, range(m_qubits), inverse=True)
    qc.measure(range(m_qubits), range(m_qubits))
    return qc

def get_var_uu_dag_circuit(centroid, x, m_qubits):
    qc = QuantumCircuit(m_qubits, m_qubits)
    c_pad = pad_and_normalize(centroid, m_qubits)
    x_pad = pad_and_normalize(x, m_qubits)
    
    qc.h(range(m_qubits))
    encode_vector(qc, c_pad, range(m_qubits))
    qc.h(range(m_qubits))
    encode_vector(qc, x_pad, range(m_qubits), inverse=True)
    qc.h(range(m_qubits))
    
    qc.measure(range(m_qubits), range(m_qubits))
    return qc

def get_uu_qnn_circuit(centroid, x, m_qubits, params):
    qc = QuantumCircuit(m_qubits, m_qubits)
    c_pad = pad_and_normalize(centroid, m_qubits)
    x_pad = pad_and_normalize(x, m_qubits)
    
    encode_vector(qc, c_pad, range(m_qubits))
    encode_vector(qc, x_pad, range(m_qubits), inverse=True)
    
    # Layer 1
    for j in range(m_qubits):
        qc.ry(params[j], j)
        
    for j in range(m_qubits - 1):
        qc.cx(j, j + 1)
        
    # Layer 2
    for j in range(m_qubits):
        qc.ry(params[j + m_qubits], j)
        
    qc.measure(range(m_qubits), range(m_qubits))
    return qc

def run_circuit_prob_0(qc, simulator, noise_model=None, shots=100):
    tc = transpile(qc, simulator)
    if noise_model is not None:
        result = simulator.run(tc, shots=shots, noise_model=noise_model).result()
    else:
        result = simulator.run(tc, shots=shots).result()
    
    counts = result.get_counts(tc)
    zero_state = '0' * qc.num_qubits
    return counts.get(zero_state, 0) / shots
