import numpy as np


def note_detection_with_onset_offset_regress(frame_output, onset_output, 
    onset_shift_output, offset_output, offset_shift_output, velocity_output,
    frame_threshold, 
    ):
    """Estimate onset and offset, onset shift, offset shift and velocity of 
    piano notes. 
    
    Args:
      frame_output: (frames_num,)
      onset_output: (frames_num,)
      onset_shift_output: (frames_num,)
      offset_output: (frames_num,)
      offset_shift_output: (frames_num,)
      velocity_output: (frames_num,)
      frame_threshold: float

    Returns: 
      output_tuples: list of [bgn, fin, onset_shift, offset_shift, normalized_velocity], 
      e.g., [
        [1821, 1909, 0.4749851, 0.3048533, 0.72119445], 
        [1909, 1947, 0.30730522, -0.45764327, 0.64200014], 
        ...]
    """
    output_tuples = []
    bgn = None
    frame_disappear = None
    offset_occur = None

    for i in range(onset_output.shape[0]):
        if onset_output[i] == 1:
            if bgn:
                """Consecutive onsets"""
                fin = i
                output_tuples.append([bgn, fin, onset_shift_output[bgn], 
                    offset_shift_output[fin], velocity_output[bgn]])
                frame_disappear, offset_occur = None, None
            bgn = i

        if bgn and i > bgn:
            """If onset found, then search offset"""
            if frame_output[i] <= frame_threshold and not frame_disappear:
                """Frame disappear detected"""
                frame_disappear = i

            if offset_output[i] == 1 and not offset_occur:
                """Offset detected"""
                offset_occur = i

            if frame_disappear:
                if offset_occur and offset_occur - bgn > frame_disappear - offset_occur:
                    """bgn --------- offset_occur --- frame_disappear"""
                    fin = offset_occur
                else:
                    """bgn --- offset_occur --------- frame_disappear"""
                    fin = frame_disappear
                output_tuples.append([bgn, fin, onset_shift_output[bgn], 
                    offset_shift_output[fin], velocity_output[bgn]])
                bgn, frame_disappear, offset_occur = None, None, None

            if bgn and (i - bgn >= 600 or i == onset_output.shape[0] - 1):
                """Offset not detected"""
                fin = i
                output_tuples.append([bgn, fin, onset_shift_output[bgn], 
                    offset_shift_output[fin], velocity_output[bgn]])
                bgn, frame_disappear, offset_occur = None, None, None

    # Sort pairs by onsets
    output_tuples.sort(key=lambda pair: pair[0])

    return output_tuples