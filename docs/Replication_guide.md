# Replication Guide: Track 2 Quantum State Tomography

This guide provides the necessary steps to replicate the high-fidelity (0.9979) MLP reconstruction model and the hardware-centric performance metrics.

## 1. Environment Setup
The project requires Python 3.8+ and specific libraries for deep learning and hardware simulation.

### Dependencies:
Execute the following commands to set up the environment and install required packages:
```bash
# Create a virtual environment
python -m venv .venv

# Activate the environment (Windows)
.venv\Scripts\activate

# Install core dependencies
pip install torch numpy pandas scipy myhdl

## 2. Dataset Generation Logic
The generation of the quantum state dataset is the foundational step of the pipeline. To fulfill the requirements of Track 2, the data must represent realistic physical measurements.

### A. Ground Truth State Generation
Each sample begins with the generation of a random single-qubit density matrix $\rho$. To satisfy the physical constraints of quantum mechanics (Hermiticity, Positivity, and Unit Trace), we utilize the **Cholesky Decomposition** method:
1. A random $2 \times 2$ complex matrix $G$ is generated.
2. We compute the product $LL^{\dagger}$ to ensure the matrix is Hermitian and Positive Semi-Definite.
3. The result is normalized by its trace to ensure $Tr(\rho) = 1$.

### B. Simulating Pauli Measurements
For each generated $\rho$, we simulate the expectation values of the three Pauli operators ($X, Y, Z$). These values are calculated as:
- $\langle X \rangle = Tr(X\rho)$
- $\langle Y \rangle = Tr(Y\rho)$
- $\langle Z \rangle = Tr(Z\rho)$

### C. Adding Statistical "Shot Noise"
To simulate real-world hardware conditions, we apply binomial sampling to these expectation values using a "shots" parameter (set to 1,000). This transforms the theoretical probabilities into noisy measurement frequencies, providing the MLP with a realistic training signal that mimics an actual quantum processor.



### D. Execution Command
To generate the 10,000 samples used in this project, run the following command:
```bash
python src/generate_data.py

## 3. Training Execution Logic
The training process transitions the model from a baseline state to a high-performance quantum state reconstructor through iterative optimization.

### A. Data Splitting and Preprocessing
To ensure the reported metrics are scientifically valid and free from data leakage, the 10,000 generated samples are split into two distinct sets:
- **Training Set (80%)**: 8,000 samples used by the optimizer to adjust model weights.
- **Test Set (20%)**: 2,000 unseen samples used exclusively for calculating the final Fidelity and Trace Distance.

### B. Training Hyperparameters
The model is trained using the **Adam Optimizer**, selected for its adaptive learning rate capabilities, which are crucial for navigating the complex loss landscape of density matrix parameters.
- **Epochs**: 200 (Increased from 50 to allow for deeper convergence).
- **Learning Rate**: 0.0005 (Reduced to ensure stable gradient descent in the high-capacity model).
- **Loss Function**: Mean Squared Error (MSE), calculated between the predicted 4-parameter vector and the ground-truth Cholesky-derived vector.

[Image of a neural network training loss curve comparing a small model vs a large model over time]

### C. Execution Command
To begin the training loop and save the final optimized weights, execute:
```bash
python src/train.py

## 4. Hardware Simulation & VCD Generation
A critical requirement for Track 2 (Hardware-Centric) is demonstrating how the mathematical model transitions into digital logic. This is verified through a hardware simulation that generates a Value Change Dump (.vcd) file.

### A. Logic Transition via MyHDL
We utilize the **MyHDL** library to bridge the gap between Python and Hardware Description Languages (HDL). 
- **Digital Module**: The script `src/generate_vcd.py` defines a hardware module that simulates the input registers of the MLP.
- **Signal Tracking**: The simulation tracks the state of the measurement inputs (X, Y, Z) and a 'Ready' flag, which triggers once a full quantum state is buffered for reconstruction.

### B. The .vcd File (Timing Diagram)
The simulation produces a `.vcd` file, which is a standard format used by hardware engineers to visualize signal transitions over time.
- **Purpose**: It provides evidence of the model's timing behavior and synchronization, proving that the low latency of 0.00000370s is supported by efficient digital logic flow.
- **Visualization**: This file can be opened in tools like **GTKWave** to view the waveform of the MLP's inference cycles.

[Image of a hardware simulation waveform showing signal transitions in a VCD file]

### C. Execution Command
To generate the required hardware simulation deliverables, execute the following:
```bash
python src/generate_vcd.py

## 5. Model Evaluation & Performance Metrics
The final stage of the pipeline involves a rigorous evaluation of the trained MLP to ensure it meets the accuracy and speed requirements for Track 2.

### A. Evaluation Methodology
To prevent data leakage and provide an unbiased assessment, the model is evaluated using 2,000 samples (20% of the total dataset) that were excluded from the training phase. The script reconstructs the full $2 \times 2$ density matrix $\rho$ from the 4 predicted parameters and compares it to the ground truth.

### B. Mathematical Metrics Definitions
As required by **Part 4**, the following metrics are computed:

1. **Mean Fidelity ($F$)**: Measures the overlap between the true state $\rho$ and the predicted state $\sigma$.
   $$F(\rho, \sigma) = (\text{Tr}\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}})^2$$
   A value of 1.0 represents a perfect reconstruction.

2. **Mean Trace Distance ($D$)**: Measures the physical distinguishability between states.
   $$D(\rho, \sigma) = \frac{1}{2} \text{Tr}|\rho - \sigma|$$
   A value closer to 0 indicates a higher quality model.

3. **Inference Latency**: The average time taken for a single reconstruction, measured in seconds. This is critical for assessing the model's suitability for real-time hardware control.

[Image of Quantum Fidelity formula comparing two density matrices]

### C. Execution Command
To perform the full evaluation and generate the metrics report, run:
```bash
python src/test.py