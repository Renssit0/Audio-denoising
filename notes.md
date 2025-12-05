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
initial learning rate 0.0001

MSE + L1 regularization

Cyclical LR + Early Stopper
base_lr
max_lr
step_size

Conv with padding=[same]!!, Conv and Upsample size=2??, [mode=linear,bilinear]!!
(https://arxiv.org/pdf/2010.14356, https://arxiv.org/pdf/2111.11773)

Conv1D
