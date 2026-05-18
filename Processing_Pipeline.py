for Participant in Participants:
  """ Reading data for the participnat"""
  Data_EEG, Labels = ReadEEGData(Participant, data_type)
  Data_NIRS = ReadNIRSData(Participant, data_type)

  """Applying Filter"""

  Filtered_Data = ApplyBandPass(Data_EEG)

  """ Framing EEG Data"""
  fs = 200
  Framed_Data = FramedData(Filtered_Data, 200)

  """ NIRS Features"""
  Framed_NIRS_Data = FramedData(Data_NIRS, 10)
  Features_NIRS_Reshaped = Framed_NIRS_Data

  if Participant == 'VP001':
    EEG_All_Participants = Framed_Data
    Labels_All_Participnats = Labels
    NIRS_All_Participants = Features_NIRS_Reshaped

  else:
    EEG_All_Participants = np.concatenate((EEG_All_Participants, Framed_Data), axis = 0)
    Labels_All_Participnats = np.concatenate((Labels_All_Participnats, Labels), axis = 0)
    NIRS_All_Participants = np.concatenate((NIRS_All_Participants, Features_NIRS_Reshaped), axis = 0)

  print(np.shape(EEG_All_Participants))
  print(np.shape(NIRS_All_Participants))


Features_EEG_Data =  extract_features(EEG_All_Participants, fs)
Features_EEG_Reshaped = np.asarray(Features_EEG_Data).astype('float32')
Features_NIRS_Reshaped = NIRS_Features(NIRS_All_Participants)