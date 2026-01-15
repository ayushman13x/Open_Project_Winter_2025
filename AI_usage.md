# AI Usage and Attribution

## 1. Disclosure
I utilized Gemini to assist in the architectural design, debugging, and documentation of this project.

## 2. Specific Prompts & Assistance
The following specific tasks were performed with AI assistance:
- **Mathematical Scaffolding**: Provided templates for the Cholesky decomposition to enforce density matrix physicality.
- **Hardware Simulation**: Generated the `myhdl` script logic to produce the required `.vcd` files for Track 2 compliance.
- **Optimization Strategy**: Assisted in iterating the MLP architecture from a baseline 64-neuron model to the high-capacity 128-neuron model.

## 3. Human Verification & Corrections
To ensure the integrity of the project and avoid "cheating," I performed the following critical verification steps:
- **Data Integrity Check**: I identified and corrected a potential data leakage issue where the AI initially suggested a test set that overlapped with training data. I enforced a strict 80/20 manual split using `.iloc` to ensure metrics were calculated on unseen data.
- **Physics Validation**: I manually verified that the final Mean Fidelity (0.9979) and Trace Distance (0.0260) were mathematically consistent for a single-qubit system.
- **Environment Debugging**: I handled the local environment configuration, resolving several `ModuleNotFoundError` issues (e.g., `myhdl` and `scipy`) that the AI could not see.
- **Metric Auditing**: I independently verified the inference latency (3.7 microseconds) by running the timing loops on my own hardware to ensure they met Track 2 performance targets.