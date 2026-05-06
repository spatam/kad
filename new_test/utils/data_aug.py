import numpy as np
import librosa as librs
import soundfile as sf
import os


def stn_std(audio, noise_factor_dB):
    signal_power = np.mean(audio**2)
    stn_lin = 10 ** (noise_factor_dB / 10)
    return np.sqrt(signal_power / stn_lin)


def data_augmentation(
    indir,
    outdir,
    aug_per_file,
    mean=0,
    noise_factor_low=15,
    noise_factor_high=5,
    scale_range=(0.5, 1.2),
    shift_factor=0.4,
):
    file_list = os.listdir(indir)

    for file in file_list:
        if file.endswith(".wav"):
            path = os.path.join(indir, file)
            audio, sr = librs.load(path, sr=None)
            for i in range(aug_per_file):
                audio_len = len(audio)
                noise_factor = np.random.randint(
                    low=noise_factor_high, high=noise_factor_low
                )
                std = stn_std(audio, noise_factor)
                noise = np.random.normal(loc=mean, scale=std, size=audio.shape)
                augmented_audio = audio + noise

                shift = np.random.randint(int(audio_len * -shift_factor), int(audio_len * shift_factor))
                scale = np.random.uniform(scale_range[0], scale_range[1]) 

                if shift > 0:
                    augmented_audio = np.pad(augmented_audio, (shift, 0), mode='constant')[:audio_len]
                elif shift < 0:
                    augmented_audio = np.pad(augmented_audio, (0, -shift), mode='constant')[-audio_len:]

                scale = np.random.uniform(*scale_range)
                augmented_audio *= scale

                augmented_audio = np.clip(augmented_audio, -1.0, 1.0)

                audio_name = file[:-4]
                filename = f"aug_{audio_name}_{i+1}.wav"

                outpath = os.path.join(outdir, filename)
                sf.write(outpath, augmented_audio, sr)

            print(
                f"File {file} correctly augmented by {aug_per_file} times and saved to {outdir}"
            )

        else:
            print(f"Skipping non-WAV file: {file}")
