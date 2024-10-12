import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

# Step 1: Load the .wav file
sample_rate, data = wavfile.read('c4.wav')

# If the data is stereo, take only one channel
if len(data.shape) > 1:
    data = data[:, 0]

# Step 2: Perform the FFT
N = len(data)  # Number of samples
fft_values = np.fft.fft(data)
fft_magnitude = np.abs(fft_values)  # Get magnitude of the FFT

# Step 3: Create frequency axis
frequencies = np.fft.fftfreq(N, 1 / sample_rate)

# Step 4: Find the first harmonic
# Ignore the zero frequency (DC component) and focus on positive frequencies
positive_frequencies = frequencies[:N // 2]
positive_magnitude = fft_magnitude[:N // 2]

# Find the frequency with the highest magnitude, excluding the DC component
index_of_first_harmonic = np.argmax(positive_magnitude[1:]) + 1  # Avoid the zero frequency
first_harmonic_freq = positive_frequencies[index_of_first_harmonic]

# Step 5: Find the lowest non-zero frequency peak
threshold = np.max(positive_magnitude) * 0.1  # Use a threshold to avoid tiny peaks
peaks = np.where(positive_magnitude > threshold)[0]  # Find indices of peaks above the threshold
lowest_frequency_peak = positive_frequencies[peaks[0]] if len(peaks) > 0 else 0  # First peak above threshold

# Step 6: Plot the FFT, highlighting both the first harmonic and the lowest frequency peak
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)  # First plot for FFT
plt.plot(positive_frequencies, positive_magnitude)
plt.axvline(x=first_harmonic_freq, color='r', linestyle='--', label=f'First Harmonic: {first_harmonic_freq:.2f} Hz')
plt.axvline(x=lowest_frequency_peak, color='g', linestyle='--', label=f'Lowest Frequency Peak: {lowest_frequency_peak:.2f} Hz')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('FFT of the Sound File')
plt.legend()
plt.grid(True)

# Step 7: Compute and plot the spectrogram
window_size = 512
overlap = 64
frequencies_spec, times, Sxx = spectrogram(data, fs=sample_rate, window='hann', nperseg=window_size, noverlap=overlap)

plt.subplot(2, 1, 2)  # Second plot for Spectrogram
plt.pcolormesh(times, frequencies_spec, 10 * np.log10(Sxx), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title('Spectrogram of the Sound File')
plt.colorbar(label='Intensity [dB]')
plt.ylim(0, sample_rate // 2)  # Limit frequency to Nyquist (half the sampling rate)

plt.tight_layout()
plt.show()

# Output the first harmonic and lowest frequency peak
print(f'The first harmonic frequency is: {first_harmonic_freq:.2f} Hz')
print(f'The lowest frequency peak is: {lowest_frequency_peak:.2f} Hz')
