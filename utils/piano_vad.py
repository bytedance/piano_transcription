import numpy as np


def note_detection_with_onset(frame_output, onset_output, threshold):
    """Estimate onset and offset pairs of notes. onset_ouput is used to detect 
    the presence of notes. frame_output is used to detect the offset of notes.
    
    Args:
      frame_output: (frames_num,)
      onset_output: (frames_num,)
      threshold: float
    
    Returns: 
      bgn_fin_pairs: list of [bgn, fin], e.g. [[9786, 9810], [11522, 11529]]
    """
    bgn_fin_pairs = []

    loct = None
    for i in range(onset_output.shape[0]):
        # onset_output is used to detect the presence of notes
        if onset_output[i] > threshold:
            if loct:
                bgn_fin_pairs.append([loct, i])
            loct = i
        if loct and i > loct:
            # frame_output is used to detect the offset of notes
            if frame_output[i] <= threshold:
                bgn_fin_pairs.append([loct, i])
                loct = None

    bgn_fin_pairs.sort(key=lambda pair: pair[0])

    return bgn_fin_pairs