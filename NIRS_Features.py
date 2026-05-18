def NIRS_BandPowers(nirs_signal):
  fs = 10
  freq_range = [0, 0.2]  # Frequency range of interest in Hz
  sampling_freq = 10  # Sampling frequency in Hz

  Features = []
  for i_files in range(np.shape(nirs_signal)[0]):
    for i_frame in range(np.shape(nirs_signal[i_files])[0]):
      for i_channels in range(72):
        data_channel = np.array(nirs_signal[i_files])[i_frame,:,i_channels]
        sample_data = np.reshape(data_channel, [np.shape(data_channel)[0], 1])

        frequencies, psd = signal.welch(sample_data, fs=sampling_freq, axis=0)
        start_index = np.argmax(frequencies >= freq_range[0])
        end_index = np.argmax(frequencies >= freq_range[1])
        psd_normalized = psd / np.sum(psd[start_index:end_index], axis=0)


        # Median
        median_bin = np.median(sample_data.squeeze(), axis=0)

        # Standard Deviation
        std_bin = np.std(sample_data.squeeze(), axis=0)

        # Root mean square (RMS)
        features_rms = np.sqrt(np.mean(np.square(sample_data)))

        # Variance
        features_variance = np.var(sample_data)

        # Hjorth parameters
        first_diff = np.diff(sample_data)
        second_diff = np.diff(first_diff)
        activity = np.var(sample_data)
        mobility = np.sqrt(np.var(first_diff) / activity)
        complexity = np.sqrt(np.var(second_diff) / np.var(first_diff))
        features_hjorth_activity = activity

        # Zero-crossing rate
        zero_crossings = librosa.feature.zero_crossing_rate(sample_data)[0]
        features_zerocrossing = np.mean(zero_crossings)

        # Peaks
        peaks, _ = scipy.signal.find_peaks(sample_data.squeeze())
        avg_peaks_time = np.mean((peaks[1:]-peaks[0:-1])/fs)
        try:
            peak_to_peak_amplitude = np.max(sample_data.squeeze()[peaks]) - np.min(sample_data.squeeze()[peaks])
        except:
            peak_to_peak_amplitude = 0

        # Area under the curve
        area = np.trapz(sample_data.squeeze())

        # Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=sample_data.squeeze(), sr=fs)[0]

        # Spectral Roll-Off
        spectral_rolloff = librosa.feature.spectral_rolloff(y=sample_data.squeeze(), sr=fs)[0]

        Outputs = np.array([median_bin, std_bin, features_rms, features_variance, features_hjorth_activity, features_zerocrossing,
                            peak_to_peak_amplitude, area, spectral_centroid[0], spectral_rolloff[0]])


        Outputs = np.reshape(Outputs, [1, np.shape(Outputs)[0]])

        if i_channels == 0:
          Channels_Select = Outputs
        else:
          Channels_Select = np.concatenate((Channels_Select, Outputs), axis = 1)

      if i_frame == 0:
        Features_Frames = Channels_Select
      else:
        Features_Frames = np.concatenate((Features_Frames, Channels_Select), axis = 0)

    Features.append(Features_Frames)

  return Features