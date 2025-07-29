import mne
import matplotlib.pyplot as plt
import os

# Define the path to your EDF file
edf_path = 'eeg_data/complete_dataset/p001_edf01.edf' 

# Create an output directory for visualizations if it doesn't exist
output_dir = 'eeg_visualizations'
os.makedirs(output_dir, exist_ok=True)

# Load the EDF file
raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
print(raw.info)

# --- 1. Plot Raw EEG and save ---
print("Plotting Raw EEG...")
fig_raw = raw.plot(n_channels=20, scalings='auto', title='Raw EEG', show=False, block=False)
fig_raw.savefig(os.path.join(output_dir, 'raw_eeg_plot.png'))
plt.close(fig_raw) # Close the figure to free memory
print(f"Saved Raw EEG plot to {os.path.join(output_dir, 'raw_eeg_plot.png')}")


# --- 2. Plot Power Spectral Density and save ---
print("Plotting Power Spectral Density...")
fig_psd = raw.plot_psd(fmax=50, show=False)
fig_psd.savefig(os.path.join(output_dir, 'psd_plot.png'))
plt.close(fig_psd) # Close the figure
print(f"Saved PSD plot to {os.path.join(output_dir, 'psd_plot.png')}")


# === Optional: Plot filtered data and save ===
# Uncomment this section to include filtered plots
# print("Plotting Filtered EEG...")
# raw_filtered = raw.copy().filter(l_freq=0.5, h_freq=30, verbose=False)
# fig_filtered = raw_filtered.plot(n_channels=20, scalings='auto', title='Filtered EEG (0.5-30Hz)', show=False, block=False)
# fig_filtered.savefig(os.path.join(output_dir, 'filtered_eeg_plot.png'))
# plt.close(fig_filtered)
# print(f"Saved Filtered EEG plot to {os.path.join(output_dir, 'filtered_eeg_plot.png')}")

print("\nAll requested visualizations saved.")
