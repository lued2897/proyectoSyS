import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
import sounddevice as sd

print(sd.query_devices())

opcion = int(input("Selecciona una opcion:\n1. Ejecutar Parte 1 del proyecto\n2. Ejecutar parte 2 del proyecto\n3.Salir\n"))

if opcion == 1:
    # Parte 1: Grabación de audio
    duration = 5  # Duration in seconds
    sample_rate = 44100  # Sample rate in Hz
    input("Presiona Enter para comenzar a grabar...")
    print("Grabando...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float64', device=None)
    sd.wait()  # Wait until recording is finished
    print("Grabación terminada")

    # Acondicionamiento de la señal
    data = audio_data[:, 0]  # Convertir a una señal unidimensional si es estéreo
    #data = data/np.max(np.abs(data))
    # FFT
    N = len(data)
    fft_values = np.fft.fft(data)
    fft_magnitude = np.abs(fft_values)
    frequencies = np.fft.fftfreq(N, 1 / sample_rate)

    # Ignorar la frecuencia cero y obtener las positivas
    positive_frequencies = frequencies[:N // 2]
    positive_magnitude = fft_magnitude[:N // 2]

    # Encuentra la primera componente armónica
    index_of_first_harmonic = np.argmax(positive_magnitude[1:]) + 1
    first_harmonic_freq = positive_frequencies[index_of_first_harmonic]

    # Encuentra el primer pico de frecuencia
    threshold = np.max(positive_magnitude) * 0.1
    peaks = np.where(positive_magnitude > threshold)[0]
    lowest_frequency_peak = positive_frequencies[peaks[0]] if len(peaks) > 0 else 0

    # Encuentra más armónicos (múltiplos enteros de la frecuencia fundamental)
    harmonics_freqs = []
    harmonics_mags = []
    for i in range(2, 6):  # Buscar hasta el 5to armónico
        harmonic_freq = first_harmonic_freq * i
        harmonic_idx = np.argmin(np.abs(positive_frequencies - harmonic_freq))
        harmonics_freqs.append(positive_frequencies[harmonic_idx])
        harmonics_mags.append(positive_magnitude[harmonic_idx])

    # Visualización de la FFT
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(positive_frequencies, positive_magnitude)
    plt.axvline(x=first_harmonic_freq, color=(1,0,0,0.5), linestyle='--', label=f'Primer armónico: {first_harmonic_freq:.2f} Hz')
    plt.axvline(x=lowest_frequency_peak, color=(0,1,0,0.5), linestyle='--', label=f'Pico más bajo: {lowest_frequency_peak:.2f} Hz')

    # Marcar armónicos adicionales
    for harmonic_freq in harmonics_freqs:
        plt.axvline(x=harmonic_freq, color=(0,0,1,0.5), linestyle='--', label=f'Armónico: {harmonic_freq:.2f} Hz')

    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud')
    plt.title('FFT del archivo')
    plt.legend()
    plt.grid(True)

    # Espectrograma
    window_size = 512
    overlap = 64
    frequencies_spec, times, Sxx = spectrogram(data, fs=sample_rate, window='hann', nperseg=window_size, noverlap=overlap)

    plt.subplot(2, 1, 2)
    plt.pcolormesh(times, frequencies_spec, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frecuencia [Hz]')
    plt.xlabel('Tiempo [s]')
    plt.title('Espectrograma')
    plt.colorbar(label='Intensidad [dB]')
    plt.ylim(0, sample_rate // 2)

    plt.tight_layout()
    plt.show()

    print(f'Frecuencia fundamental:: {first_harmonic_freq:.2f} Hz')
    print(f'Pico más bajo: {lowest_frequency_peak:.2f} Hz')
    for i, harmonic_freq in enumerate(harmonics_freqs, start=2):
        print(f'armónico {i}: {harmonic_freq:.2f} Hz')


elif opcion == 2:
    # Cargar el archivo .wav
    sample_rate, data = wavfile.read('archivo.wav')

    # Si el archivo es estéreo, tomar solo un canal
    if len(data.shape) > 1:
        data = data[:, 0]
    #data = data/np.max(np.abs(data))

    # FFT
    N = len(data)
    fft_values = np.fft.fft(data)
    fft_magnitude = np.abs(fft_values)

    # Frecuencias positivas
    frequencies = np.fft.fftfreq(N, 1 / sample_rate)
    positive_frequencies = frequencies[:N // 2]
    positive_magnitude = fft_magnitude[:N // 2]

    # Encuentra la primera componente armónica
    index_of_first_harmonic = np.argmax(positive_magnitude[1:]) + 1
    first_harmonic_freq = positive_frequencies[index_of_first_harmonic]

    # Encuentra el primer pico de frecuencia
    threshold = np.max(positive_magnitude) * 0.1
    peaks = np.where(positive_magnitude > threshold)[0]
    lowest_frequency_peak = positive_frequencies[peaks[0]] if len(peaks) > 0 else 0

    # Encuentra más armónicos (múltiplos enteros de la frecuencia fundamental)
    harmonics_freqs = []
    harmonics_mags = []
    for i in range(2, 6):  # Buscar hasta el 5to armónico
        harmonic_freq = first_harmonic_freq * i
        harmonic_idx = np.argmin(np.abs(positive_frequencies - harmonic_freq))
        harmonics_freqs.append(positive_frequencies[harmonic_idx])
        harmonics_mags.append(positive_magnitude[harmonic_idx])

    # Visualización de la FFT
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(positive_frequencies, positive_magnitude)
    plt.axvline(x=first_harmonic_freq, color=(1,0,0,0.5), linestyle='--', label=f'Primer armónico: {first_harmonic_freq:.2f} Hz')
    plt.axvline(x=lowest_frequency_peak, color=(0,1,0,0.5), linestyle='--', label=f'Pico más bajo: {lowest_frequency_peak:.2f} Hz')

    # Marcar armónicos adicionales
    for harmonic_freq in harmonics_freqs:
        plt.axvline(x=harmonic_freq, color=(0,0,1,0.5), linestyle='--', label=f'Armónico: {harmonic_freq:.2f} Hz')

    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud')
    plt.title('FFT del archivo')
    plt.legend()
    plt.grid(True)

    # Espectrograma
    window_size = 512
    overlap = 64
    frequencies_spec, times, Sxx = spectrogram(data, fs=sample_rate, window='hann', nperseg=window_size, noverlap=overlap)

    plt.subplot(2, 1, 2)
    plt.pcolormesh(times, frequencies_spec, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frecuencia (Hz)')
    plt.xlabel('Tiempo [s]')
    plt.title('Espectrograma del archivo')
    plt.colorbar(label='Intensidad [dB]')
    plt.ylim(0, sample_rate // 2)

    plt.tight_layout()
    plt.show()

    print(f'The first harmonic frequency is: {first_harmonic_freq:.2f} Hz')
    print(f'The lowest frequency peak is: {lowest_frequency_peak:.2f} Hz')
    for i, harmonic_freq in enumerate(harmonics_freqs, start=2):
        print(f'Harmonic {i}: {harmonic_freq:.2f} Hz')

else:
    print("Adios")

