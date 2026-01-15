import numpy as np
import pandas as pd
import os

def generate_physical_rho():
    # Generating a random complex matrix to start the Cholesky process
    L = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
    
    # Filtering the matrix to keep only the lower triangular part
    L = np.tril(L)
    
    # Computing the density matrix using the LL^dagger formula to ensure physicality
    rho_unnormalized = L @ L.conj().T
    
    # Normalizing by the trace so that the total probability equals 1
    return rho_unnormalized / np.trace(rho_unnormalized)

def simulate_measurements(rho, shots=1000):
    # Setting up the Z-basis projectors (Computational basis)
    Pz0 = np.array([[1, 0], [0, 0]])
    
    # Defining X-basis projectors for the |+> and |-> states
    Px0 = 0.5 * np.array([[1, 1], [1, 1]])
    
    # Defining Y-basis projectors for the |+i> and |-i> states
    Py0 = 0.5 * np.array([[1, -1j], [1j, 1]])

    # Applying the Born Rule to calculate probabilities for all bases
    prob_z0 = np.real(np.trace(Pz0 @ rho))
    prob_x0 = np.real(np.trace(Px0 @ rho))
    prob_y0 = np.real(np.trace(Py0 @ rho))

    # Simulating the random measurement counts based on calculated probabilities
    # Binomial sampling reflects the statistical noise present in real hardware
    count_z0 = np.random.binomial(shots, prob_z0)
    count_x0 = np.random.binomial(shots, prob_x0)
    count_y0 = np.random.binomial(shots, prob_y0)

    # Returning the normalized frequencies (counts/shots) as a feature vector
    # This list [x_freq, y_freq, z_freq] is the input our Neural Network will study
    return [count_x0/shots, count_y0/shots, count_z0/shots]

# Executing the generation loop for 10,000 samples
os.makedirs('data/training', exist_ok=True)
data_list = []

print("Starting data generation...")
for i in range(10000):
    rho = generate_physical_rho()
    features = simulate_measurements(rho)
    
    # Flattening rho into its components (real and imaginary) to save as "labels"
    # We only need the top-left, top-right (complex), and bottom-right values
    data_list.append(features + [rho[0,0].real, rho[1,1].real, rho[0,1].real, rho[0,1].imag])

# Saving the final dataset to a CSV file for training
df = pd.DataFrame(data_list, columns=['x', 'y', 'z', 'r00', 'r11', 'r01_real', 'r01_imag'])
df.to_csv('data/training/dataset.csv', index=False)
print("Finished saving 10,000 samples to data/training/dataset.csv")