# Audio-denoising
Uni project about audio denoising.

### TO DO 

- Dataset ✓
- Dataset Analysis ✓
- Dataloader ✓

Understand the 3 main architectures along with its hiperparameters and 3 main metrics.

Metrics
SNR PESQ
STOI?
verispeak  afdsafojsdauflgf NO
CSIG
CBAK
COVL


# Dataset 

The 28 speaker version of [CSTR VCTK - DEMAND](https://doi.org/10.7488/ds/2117) dataset was used in this project.

If you wish to replicate my findings, after downloading and extracting the zip files your `/data` directory should look something like this.

```bash
data/
├── clean_testset_wav
├── clean_trainset_28spk_wav
├── noisy_testset_wav
├── noisy_trainset_28spk_wav
├── testset_txt
└── trainset_28spk_txt
```

After which, a script in the ############## will downgrade the sample rate from 48kHz to 16kHz into a `/data_16k` folder

# References


- [A Comparative Evaluation of Deep Learning Models for Speech Enhancement in Real-World Noisy Environments](https://doi.org/10.48550/arXiv.2506.15000) - 2025
- [CMGAN: Conformer-Based Metric-GAN for Monaural Speech Enhancement](https://doi.org/10.48550/arXiv.2209.11112) - 2022-2024
- [Simultaneous Speech Denoising and Super-Resolution Using mGLFB-Based U-Net, Fine-Tuned via Perceptual Loss](https://www.mdpi.com/3474914) - 2025

- [Options for Performing DNN-Based Causal Speech Denoising Using the U-Net Architecture](https://www.mdpi.com/3067016) - 2024


- [A Survey of Audio Enhancement Algorithms for Music, Speech, Bioacoustics, Biomedical, Industrial, and Environmental Sounds by Image U-Net](https://ieeexplore.ieee.org/document/10371226) - 2024

# License

The dataset is licensed under:

- [CC BY-NC 4.0 ](http://creativecommons.org/licenses/by/4.0/)

