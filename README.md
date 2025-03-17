# kad
Keystroke Acoustic Dataset (KAD)

Welcome to the official repository for the Keystroke Acoustic Dataset (KAD) dataset, a comprehensive audio dataset designed for keystroke classification research vulnerability.
We are pleased to present a new dataset for keystroke attack research, now available on GitHub. 
This dataset has been recorded using the microphone of a smartphone and based on the methodology outlined by Harrison et al. [1].

# Desk experiment setup

This is the experiment setup for keystroke with a mechanical keyboard.
The smartphone is at 17cm from the keyboard position.

![image](GA.jpg)

# Dataset Overview

The first dataset includes audio recordings of keystrokes from a MacBook Pro keyboard and is publicly accessible on GitHub [2]. 
Building upon this, one additional dataset were created at typing speed of 1 sec and with different energy to simulate diverse real-world scenarios. 
This datasets contains 36 audio files. 
After audio splitting, you must have 900 audio files with 75 .wav files for each of the 36 keys (letters a-z and numbers 0-9). 
After applying data augmentation techniques, the number of audio files in each dataset expands to 2,700.

In this repository there is only the mechanical dataset, recorded with the above desk setup.
For segmentation, each audio file was processed to isolate keystroke sounds, with the start of the segment aligned to the peak time and the end set one second before the click sound, as determined experimentally.

This extensive dataset supports the evaluation of various models and techniques for detecting and analyzing keystroke sounds under realistic conditions. Details, including the folder organization, are available into the dataset.
Finally, it is important to emphasize that the presented dataset is by no means intended to promote or encourage attacks of this kind, but rather to highlight a vulnerability often overlooked in the cybersecurity landscape.

# Source code
For security reasons, the source code can be requested only by sending a direct request to my email address: massimo.spata@unict.it. We will evaluate each individual request.

# Ethical use of Kad Dataset
The Kad dataset is intended solely for research purposes to highlight a newly discovered vulnerability. Ethical use requires that it not be exploited for malicious activities but rather to improve cybersecurity defenses. Researchers must adhere to ethical guidelines, ensuring responsible handling of the data and transparency in their findings. Any use should align with legal and ethical standards, fostering a safer digital environment by mitigating potential threats rather than enabling exploitation.

# Citation
If you use the kad dataset in your research, please provide proper citation:

M. O. Spata, V. M. Russo, A. Ortis and S. Battiato, "A New Pipeline for Snooping Keystroke Based on Deep Learning Algorithm," in IEEE Access, doi: 10.1109/ACCESS.2025.3536877.

M. O. Spata, V. M. Russo, A. Ortis and S. Battiato, "Acoustic Side Channel Attack for Keystroke Splitting in the Wild," 2024 IEEE International Conference on Metrology for eXtended Reality, Artificial Intelligence and Neural Engineering (MetroXRAINE), St Albans, United Kingdom, 2024, pp. 131-136, doi: 10.1109/MetroXRAINE62247.2024.10796234.

M.O. Spata, V.M. Russo, A. Ortis, S. Battiato (2024). A New Deep Learning Pipeline for Acoustic Attack on Keyboards. In: Arai, K. (eds) Intelligent Systems and Applications. IntelliSys 2024. Lecture Notes in Networks and Systems, vol 1065. Springer, Cham. https://doi.org/10.1007/978-3-031-66329-1_26

Kad DOI: 10.5281/zenodo.14809537

# References

[1] J. Harrison, E. Toreini and M. Mehrnezhad, "A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards," in 2023 IEEE European Symposium on Security and Privacy Workshops (EuroS&PW), Delft, Netherlands, (2023) pp. 270-280.

[2] J. Harrison, et al. Keystroke-Datasets https://github.com/JBFH-Dev/Keystroke-Datasets (2023)

# Acknowledgments
Thank you to all participants for their contributions and to the team for their efforts in data collection and processing.

We look forward to seeing how this dataset advances the field of keystroke acoustic attacks vulnerability!
