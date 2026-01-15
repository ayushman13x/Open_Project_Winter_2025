import torch
import numpy as np
import pandas as pd
import time
from model import QST_Model

def calculate_metrics(rho_true, rho_pred):
    """
    Calculates exact Fidelity and Trace Distance for 2x2 density matrices.
    """
    # 1. Trace Distance: 0.5 * Tr|rho_true - rho_pred|
    diff = rho_true - rho_pred
    # Absolute eigenvalues of the difference matrix
    eigvals = np.linalg.eigvals(diff)
    trace_dist = 0.5 * np.sum(np.abs(eigvals))

    # 2. Fidelity: (Tr(sqrt(sqrt(rho_true) * rho_pred * sqrt(rho_true))))^2
    # For a single qubit, this simplifies to: Tr(rho_true @ rho_pred) + 2*sqrt(det(rho_true)*det(rho_pred))
    # We use the robust version for mixed states:
    from scipy.linalg import sqrtm
    sq_rho_true = sqrtm(rho_true)
    fidelity_mat = sq_rho_true @ rho_pred @ sq_rho_true
    fid = np.real(np.trace(sqrtm(fidelity_mat)))**2
    
    return np.real(fid), np.real(trace_dist)

# 1. Loading the trained 'student' model
model = QST_Model()
model.load_state_dict(torch.load('outputs/model_weights.pt'))
model.eval()

# 2. Loading the dataset to get test samples (the last 1000 rows)
df = pd.read_csv('data/training/dataset.csv')
test_df = df.iloc[8000:] # Using unseen data from the end of the file

X_test = torch.tensor(test_df[['x', 'y', 'z']].values, dtype=torch.float32)

# 3. Measuring Inference Latency (Part 4.2)
start_time = time.time()
with torch.no_grad():
    predictions = model(X_test).numpy()
end_time = time.time()

total_time = end_time - start_time
latency = total_time / 1000

# 4. Calculating Accuracy Metrics (Part 4.1)
fidelities = []
trace_distances = []

for i in range(len(predictions)):
    # Target (True) Rho from dataset
    t = test_df.iloc[i]
    rho_true = np.array([[t['r00'], complex(t['r01_real'], -t['r01_imag'])],
                         [complex(t['r01_real'], t['r01_imag']), t['r11']]])
    
    # Predicted Rho from Model Outputs
    p = predictions[i]
    # Reconstructing the matrix elements predicted by the MLP
    rho_pred = np.array([[p[0], complex(p[2], -p[3])],
                         [complex(p[2], p[3]), p[1]]])
    
    # Ensure prediction is normalized (Tr=1) for fair metric calculation
    rho_pred = rho_pred / np.trace(rho_pred)
    
    fid, td = calculate_metrics(rho_true, rho_pred)
    fidelities.append(fid)
    trace_distances.append(td)

# 5. Reporting the Final Results
print("-" * 30)
print("FINAL EVALUATION REPORT")
print("-" * 30)
print(f"Mean Fidelity:       {np.mean(fidelities):.6f}")
print(f"Mean Trace Distance: {np.mean(trace_distances):.6f}")
print(f"Inference Latency:   {latency:.8f} seconds per state")
print("-" * 30)