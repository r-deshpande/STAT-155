#Importing the necessary libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tqdm import tqdm

# Loading my data
data = pd.read_csv('/Users/ruchadesh/Documents/Documents - Rucha’s MacBook Air/GitHub/STAT-155/Project_1/Data/detectors_prepped_with_document.csv')

# This line of code creates the binary response variable - 0 if not misclassified and 1 if misclassified by the predictor AI
data['response'] = np.abs(data['kind'].astype('category').cat.codes -
                          data['.pred_class'].astype('category').cat.codes)

# Native represents whether a speaker is a native English speaker or not
# Recodes the column of data to be binary: 0 if non-native, 1 if native
data['native'] = data['native'].astype('category').cat.codes

print(data['native'])

# Extracting coefficients
X = sm.add_constant(data['native'])
y = data['response']
original_model = sm.Logit(y, X).fit()
print(original_model.summary())

# Parameters
beta_0, beta_1 = original_model.params

# Monte Carlo Simulation Setup
NUM_SIMULATIONS = 1000
N = len(data)

# Store simulation results
beta_1_estimates = []

# Run simulations
for _ in tqdm(range(NUM_SIMULATIONS)):
    # Simulate native variable from empirical distribution
    native_sim = np.random.choice(data['native'], size=N, replace=True)

    # Generate simulated probabilities and responses
    logits = beta_0 + beta_1 * native_sim
    prob = 1 / (1 + np.exp(-logits))
    response_sim = np.random.binomial(1, prob)

    # Fit logistic regression on simulated data
    X_sim = sm.add_constant(native_sim)
    model_sim = sm.Logit(response_sim, X_sim).fit(disp=False)
    
    # Save estimated beta_1
    beta_1_estimates.append(model_sim.params[1])
    native_sim = np.random.choice(data['native'], size=N, replace=True)

    # Generate simulated probabilities and responses
    logits = beta_0 + beta_1 * native_sim
    prob = 1 / (1 + np.exp(-logits))
    response_sim = np.random.binomial(1, prob)

    # Fit logistic regression on simulated data
    X_sim = sm.add_constant(native_sim)
    model_sim = sm.Logit(response_sim, X_sim).fit(disp=False)
    
    # Save estimated beta_1
    beta_1_estimates.append(model_sim.params[1])

# Saving the results and putting them in a dataframe
sim_results = pd.DataFrame({'beta_1': beta_1_estimates})

# Save results to CSV
sim_results.to_csv("project4_beta1_simulation_results.csv", index=False)

# Plot distribution of beta_1 estimates
plt.close()
plt.hist(sim_results['beta_1'], bins=30, edgecolor='black')
plt.axvline(beta_1, color='red', linestyle='--', label='Original Estimate')
plt.title('Distribution of β₁ Estimates')
plt.xlabel('β₁')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig("project4_beta1_distribution_plot.png")
plt.show()

