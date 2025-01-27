# kad
Keystroke Acoustic Dataset (KAD)

Welcome to the official repository for the Keystroke Acoustic Dataset (KAD) dataset, a comprehensive audio dataset designed for keystroke classification research vulnerability.
We are pleased to present a new dataset for keystroke attack research, now available on GitHub. 
This dataset has been recorded using the microphone of a smartphone and based on the methodology outlined by Harrison et al. [1].

# Desk experiment setup

This is the experiment setup for keystroke with a mechanical keyboard.
The smartphone is at 17cm from the keyboard position.

![image](GA.jpeg)

# Dataset Overview

The first dataset includes audio recordings of keystrokes from a MacBook Pro keyboard and is publicly accessible on GitHub [2]. 
Building upon this, one additional dataset were created at typing speed of 1 sec and with different energy to simulate diverse real-world scenarios. 
This datasets contains 36 audio files. 
After audio splitting, you must have 900 audio files with 75 .wav files for each of the 36 keys (letters a-z and numbers 0-9). 
After applying data augmentation techniques, the number of audio files in each dataset expands to 2,700.

In this repository there is only the mechanical dataset, recorded with the above desk setup.
For segmentation, each audio file was processed to isolate keystroke sounds, with the start of the segment aligned to the peak time and the end set one second before the click sound, as determined experimentally.

This extensive dataset supports the evaluation of various models and techniques for detecting and analyzing keystroke sounds under realistic conditions. Full details, including the segmentation process and folder organization, are available alongside the dataset.
Finally, it is important to emphasize that the presented dataset is by no means intended to promote or encourage attacks of this kind, but rather to highlight a vulnerability often overlooked in the cybersecurity landscape.

# Citation
If you use the kad dataset in your research, please provide proper citation.

# References

[1] J. Harrison, E. Toreini and M. Mehrnezhad, "A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards," in 2023 IEEE European Symposium on Security and Privacy Workshops (EuroS&PW), Delft, Netherlands, (2023) pp. 270-280.

[2] J. Harrison, et al. Keystroke-Datasets https://github.com/JBFH-Dev/Keystroke-Datasets (2023)

# Acknowledgments
Thank you to all participants for their contributions and to the team for their efforts in data collection and processing.

We look forward to seeing how this dataset advances the field of keystroke acoustic attacks vulnerability!
