def extract_features(eeg_signal, sampling_rate):
    Features = []
    for i_files in range(np.shape(eeg_signal)[0]):
      for i_frame in range(np.shape(eeg_signal[i_files])[0]):
        for i_channels in range(30):
          data_channel = np.array(eeg_signal[i_files])[i_frame,:,i_channels]

          sample_data = np.reshape(data_channel, [np.shape(data_channel)[0], 1])

          # Median
          median_bin = np.median(sample_data.squeeze(), axis=0)

          # Standard Deviation
          std_bin = np.std(sample_data.squeeze(), axis=0)

          # Root mean square (RMS)
          features_rms = np.sqrt(np.mean(np.square(sample_data)))

          Power_theta = bandpower(sample_data, fs, 4, 8)                  #Theta Band Power

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

          Power_alpha = bandpower(sample_data, fs, 8, 14)                 #Alpha Band Power
          Power_beta = bandpower(sample_data, fs, 14, 30)                 #Beta Band Power

          Power_alpha_theta = Power_alpha/Power_theta                     #Ratios
          Power_theta_beta = Power_theta/Power_beta                       #Ratios

          # Peaks
          peaks, _ = scipy.signal.find_peaks(sample_data.squeeze())
          avg_peaks_time = np.mean((peaks[1:]-peaks[0:-1])/fs)
          peak_to_peak_amplitude = np.max(sample_data.squeeze()[peaks]) - np.min(sample_data.squeeze()[peaks])


          Outputs = np.array([median_bin, std_bin, features_rms, Power_theta[0][0], features_variance,
                              features_hjorth_activity, features_zerocrossing, Power_alpha_theta[0][0], Power_theta_beta[0][0], peak_to_peak_amplitude])


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




""" This function is used as part of EEG Features - Bandpower calculation across different bands Alpha, Beta, etc."""

def bandpower(x, fs, fmin, fmax):
  f, Pxx_den = scipy.signal.welch(x, fs, window='hann',  axis = 0, average='mean')
  ind_min = np.argmax(f > fmin) - 1
  ind_max = np.argmax(f > fmax) - 1
  Power = []
  for i in range(np.shape(Pxx_den)[1]):
    Power.append(np.trapz(Pxx_den[ind_min: ind_max, i], f[ind_min: ind_max]))
  Power = np.array(Power)
  Power = np.reshape(Power, (np.shape(Power)[0], 1))
  return Power