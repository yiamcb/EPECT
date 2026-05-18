import numpy as np
import pandas as pd
from scipy.stats import f_oneway

# Data shapes
samples_per_subject = 1560 // 26

# Placeholder for results
anova_results = []

# Define helper function for ANOVA
def perform_anova(groups):
    """Perform ANOVA across groups."""
    return f_oneway(*groups)

# Analysis 1: Cross-subject variations (same label)
for label in [0, 1]:
    eeg_label_data = eeg_data[labels == label]
    nirs_label_data = nirs_data[labels == label]
    # Split filtered data into groups for each subject
    eeg_groups = np.array_split(eeg_label_data, 26)
    nirs_groups = np.array_split(nirs_label_data, 26)
    eeg_anova = perform_anova([group.reshape(-1) for group in eeg_groups])
    nirs_anova = perform_anova([group.reshape(-1) for group in nirs_groups])
    anova_results.append({
        "Analysis": "Cross-subject",
        "Label": label,
        "EEG F-statistic": eeg_anova.statistic,
        "EEG p-value": eeg_anova.pvalue,
        "fNIRS F-statistic": nirs_anova.statistic,
        "fNIRS p-value": nirs_anova.pvalue
    })

# Analysis 2: Variations over time (frames)
eeg_time_groups = [eeg_data[:, frame, :].reshape(-1) for frame in range(eeg_data.shape[1])]
nirs_time_groups = [nirs_data[:, frame, :].reshape(-1) for frame in range(nirs_data.shape[1])]
eeg_time_anova = perform_anova(eeg_time_groups)
nirs_time_anova = perform_anova(nirs_time_groups)
anova_results.append({
    "Analysis": "Time-variations",
    "Label": "All",
    "EEG F-statistic": eeg_time_anova.statistic,
    "EEG p-value": eeg_time_anova.pvalue,
    "fNIRS F-statistic": nirs_time_anova.statistic,
    "fNIRS p-value": nirs_time_anova.pvalue
})

# Analysis 3: Differences between 1 and 0 cases
eeg_label_1 = eeg_data[labels == 1].reshape(-1)
eeg_label_0 = eeg_data[labels == 0].reshape(-1)
nirs_label_1 = nirs_data[labels == 1].reshape(-1)
nirs_label_0 = nirs_data[labels == 0].reshape(-1)
eeg_label_anova = perform_anova([eeg_label_1, eeg_label_0])
nirs_label_anova = perform_anova([nirs_label_1, nirs_label_0])
anova_results.append({
    "Analysis": "Label differences",
    "Label": "1 vs 0",
    "EEG F-statistic": eeg_label_anova.statistic,
    "EEG p-value": eeg_label_anova.pvalue,
    "fNIRS F-statistic": nirs_label_anova.statistic,
    "fNIRS p-value": nirs_label_anova.pvalue
})

anova_table = pd.DataFrame(anova_results)


import ace_tools as tools; tools.display_dataframe_to_user(name="ANOVA Analysis Summary", dataframe=anova_table)
