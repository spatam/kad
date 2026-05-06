import numpy as np
import os
import librosa as librs
import pywt
from pydub import AudioSegment


def split_audio(file_path, peak_times, gap_time, outdir, name_no_ext = None):
    audio = AudioSegment.from_file(file_path)
    if name_no_ext is None:
        name_no_ext = os.path.splitext(os.path.basename(file_path))[0]

    for i, peak_time in enumerate(peak_times):
        start_time_ms = int(peak_time * 1000)

        if start_time_ms < 0:
            start_time_ms = 0

        segment_duration_s = gap_time - 0.1
        end_time_ms = int((peak_time + segment_duration_s) * 1000)

        split = audio[start_time_ms:end_time_ms]

        outname = os.path.join(outdir, f"{name_no_ext}_{i+1}.wav")

        split.export(outname, format="wav")


def peak_detection(file_path, wavelet_type, outdir, name = None):
    gap_time = 0.5
    y, sr = librs.load(file_path)
    coef, _ = pywt.cwt(y, np.arange(1, 128), wavelet_type)

    noise_level = np.mean(np.abs(coef[:, : int(sr)]))
    threshold = 5 * noise_level

    i = 0
    peak_times = []

    while i < coef.shape[1]:
        is_peak_detected = False
        for j in range(coef.shape[0]):

            if i > 0:
                maybe_p = abs(coef[j, i] - coef[j, i - 1])

                if maybe_p > threshold:
                    time = librs.samples_to_time(i, sr=sr)
                    peak_times.append(time)
                    i += int(gap_time * sr)
                    is_peak_detected = True
                    break

        if not is_peak_detected:
            i += 1

    split_audio(file_path, peak_times, gap_time, outdir, name)

    return f"Splitting process completed. Found {len(peak_times)} segments."
