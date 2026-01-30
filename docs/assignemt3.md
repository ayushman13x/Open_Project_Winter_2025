# Assignment 3: Scalable Tomography Report

## 1. Serialization Strategy
For this assignment, we implemented a serialization pipeline using the `pickle` library. 
- **Format**: `.pkl` files are used to store model weights, configuration, and RNG states.
- **Alternative**: We would consider **HDF5** if the model parameters scaled into the millions, as it offers better efficiency for large numerical arrays.

## 2. Scalability Study Results
Our benchmarks for 1 through 6 qubits revealed the following trends:
- **Runtime**: Classical simulation time increases exponentially ($2^n$) as the Hilbert space dimension grows.
- **Fidelity**: Mean fidelity remains stable across qubit counts in our surrogate model, but the computational cost of verifying this fidelity on classical hardware becomes a bottleneck beyond 6-8 qubits.

## 3. Ablation Study
We tested varying the number of layers (depth) in our `QuantumModel`.
- **Finding**: Increasing layers allows the model to capture more complex state correlations but introduces additional inference latency.
- **Hardware Impact**: For Track 2, we must balance model depth with real-time execution constraints.

## 4. Scaling Limits and Future Work
Current classical pipelines hit a "scaling wall" due to the exponential growth of the density matrix. 
- **Next Steps**: We recommend exploring **Classical Shadows** to estimate observables without reconstructing the full statevector, or utilizing specialized hardware accelerators to handle the $2^n$ matrix math.