FROM [1]

# DATASET DESCRIPTION

1) DEMAND Dataset: The DEMAND (Diverse Environments Multichannel Acoustic Noise Database) dataset is a
comprehensive collection of real-world noise recordings captured in various environments, including domestic, office,
and outdoor settings. Each recording uses 16 microphones to
capture the spatial characteristics, which refer to the properties
of sound or noise that describe how it behaves in a physical
space, providing a more realistic representation of acoustic
conditions [16].

3) VCTK Corpus Dataset: The VCTK Corpus is a multispeaker English speech dataset produced by the Centre for
Speech Technology Research (CSTR) at the University of
Edinburgh [18]. It consists of clean speech recordings from
110 speakers with diverse accents. In this study, we used
the VCTK dataset as a source of clean speech, which was
subsequently mixed with noise from the DEMAND dataset to
create challenging noisy conditions for training the CMGAN
model.

# TRAINING

## Data Preprocessing

TLDR: Downsampling, matching file extensions
 
## Trainig Parameters

## Experimental Setup

TLDR: We ball.

Adam optimizer, with 
initial learning rate 0.0001 (after surveying various of the cited papers)
WIH
MSE + L1 regularization or L1, hubert?

Wave U Net (https://doi.org/10.23919/FRUCT49677.2020.9211072) -> MSE, 0.0001

Cyclical LR + Early Stopper
base_lr
max_lr
step_size

Channel step size?
    wave u net: 24 (after surveying various of the cited papers)
    u net: exponential incresase


Conv with padding=[same]!!, Conv and Upsample size=2, [mode=linear,bilinear]!!
(https://arxiv.org/pdf/2010.14356, https://arxiv.org/pdf/2111.11773)

Noise artifacts:

tho Demucs-/MelGAN
[In initial tests, we also found artifacts when using such convolutions as upsampling blocks in our Wave-U-Net model in the form of high-frequency buzzing noise.](https://arxiv.org/pdf/1806.03185) be faithful to the og just go for linear interpolation and no trans

[Interpolation upsamplers, that can introduce filtering artifacts](https://arxiv.org/pdf/2010.14356)

tonal artifacts (no periodic tones), but they do introduce filtering artifacts (attenuation/colouring of high frequencies).

As a solution, we employ convolutions without implicit
padding and instead provide a mixture input larger than
the size of the output prediction

# Math regarding the channel sizes

depth = 8
core = 880000
remainder = 0 # such that its equal or greater than 0 and divisible by 2
# must fine tune so the final result after the forward is greater or equal to core

tensor = remainder + core

tensor = tensor + 4
tensor = math.ceil(tensor/2)

for _ in range(depth-1):
    tensor = tensor  + 4 # conv
    tensor = math.ceil((tensor+1)/2 )# interpolation

tensor = tensor + 14

for _ in range(depth):
    tensor = 2*tensor # decimate
    tensor = tensor + 14 # conv

print('Results')
print(tensor)
padding = (tensor - core)/2
print(padding) # must be int
print()
# padding -= 1727 # how do i find this?


# forward
n = 880000 + 2*padding

skips = []
for i in range(depth):
    print(n)
    n = n-14 # conv
    skips.append(n)
    n = math.ceil(n / 2.0)  # decimate

n = n - 14 # mid conv

print(skips)
print()

for i in range(depth-1,0,-1):
    n = 2*n - 1 # interpolation
    n = n-4 # conv
    print(skips[i])
    print(n)
    print()

# extra
n = 2*n # interpolation
n = n-4 # conv
print(skips[0])
print(n)
print()

n = n # final conv
print(n)

Results
888562
4281.0

888562.0
444274
222130
111058
55522
27754
13870
6928
[888548.0, 444260, 222116, 111044, 55508, 27740, 13856, 6914]

6914
6881

13856
13757

27740
27509

55508
55013

111044
110021

222116
220037

444260
440069

888548.0
880134

880134


However, it's important to note that Nustede's 2021 paper was not the first to apply U-Net to audio tasks in general. The timeline shows:​

First U-Net Applications to Audio
Jansson et al. (2017) - "Singing Voice Separation with Deep U-Net Convolutional Networks" at ISMIR 2017, was one of the pioneering applications of U-Net to audio, specifically for separating vocals from instrumental backing tracks.​

Stoller et al. (2018) - "Wave-U-Net" at ISMIR 2018, introduced a 1D time-domain adaptation of U-Net for end-to-end audio source separation.​

Nustede and Anemüller (2021) - Your first paper , introduced the variational U-Net for speech enhancement, which appears to be among the first to apply U-Net specifically to speech denoising rather than music source separation.​

The other papers you mentioned ( Baloch 2023 and Hossain 2023) both came after Nustede's work. So while Jansson pioneered U-Net in audio processing for music separation in 2017, Nustede 2021 appears to be the first among your listed references and likely one of the earliest to apply it specifically to speech enhancement/denoising tasks

============================================================
Training: Wave_UNet with L1Loss for 200 epochs
Epoch 010 | train_loss=0.0082 | val_loss=0.0081
Epoch 020 | train_loss=0.0071 | val_loss=0.0074
Epoch 030 | train_loss=0.0065 | val_loss=0.0070
Epoch 040 | train_loss=0.0061 | val_loss=0.0066
Epoch 050 | train_loss=0.0057 | val_loss=0.0066
Epoch 060 | train_loss=0.0055 | val_loss=0.0065
Epoch 070 | train_loss=0.0052 | val_loss=0.0065
Epoch 080 | train_loss=0.0051 | val_loss=0.0065

made me consider min delta to avoid overfitting


Section “Network Description” explains that DVUNET is a U‑Net–style encoder–decoder with dilated convolutions and a variational bottleneck.​

Later, the authors state that “taking away the variational bottleneck from the DVUNET is abbreviated DUNET” and “further removing the dilated convolutions results in the UNET.”




For 2D spectrograms, several works use transposed convs in U‑Net‑style decoders without reporting severe artifacts, because:

The spectrogram is already a heavily smoothed, band‑limited representation.

Any checkerboard pattern in the spectrogram often translates into relatively mild amplitude ripples rather than distinct buzzing in the waveform, especially when the noisy phase is reused.​​

https://source-separation.github.io/tutorial/approaches/deep/architectures.html
https://mac.kaist.ac.kr/~juhan/gct634/2021-Fall/Slides/%5Bweek11-1%5D%20AE,%20U-Net,%20and%20source%20separation.pdf