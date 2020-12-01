# Parallel WaveNet

link

## Pre-requisites

## Key Ideas

## Main Body

### Why Parallelize?

The sequential generation of audio samples due to the autoregressive nature of WaveNet is too slow for production environments. To solve this, a method called **probability density distillation** was used to generate long sequences of samples all at once and with **no loss of quality** over the original WaveNet. A fully-trained WaveNet model to **teach a second, “student” network** that is smaller dilated convolutional neural network, similar to the original WaveNet. Generation of each sample in the student network **does not** depend on any of the previously generated samples, meaning we can generate the first and last word - and everything in between parallelly making this method a lot more efficient. 

### WaveNet: Some brief info + changes

Autoregressive networks are modelled by this by probability distribution -sw

![](/home/kritianandan/Pictures/autoreg.png)

Due to the convolutional structure of WaveNet, it can be parallelized during training time (as all future audio samples are available), but during inference the audio samples need to be generated sequentially (x<sub>t</sub> before x<sub>(t-1)</sub>).

To model high dimensional data such as audio one would require a network with a broader receptive field (filter size for convolutional networks) resulting in a very deep network capture such dependencies. WaveNet tackled this by using **dilations** (skipped weights in between by filling zeros in the filter). The authors did try to reduce the size of the WaveNet network to deploy it in real-time but that resulted in degraded quality.

In this paper, the authors increased the fidelity of WaveNet by making some changes. They used 16-bit audio instead of 8 bit audio. Instead of using mu-law they used discretized mixture of logistics distribution, as modelling 65,536-way categorical distribution would be very costly. They increased the sampling rate from 16kHz to 24kHz and the dilation factor from 2 to 3 to cope with the increased dimension in audio data.

### Inverse-Autoregressive Flows (IAF)

A different architecture to support rapid, parallel generation of audio samples was needed.

A Normalizing Flow is a transformation of a simple probability distribution (e.g., a standard normal) into a more complex distribution by a sequence of invertible and differentiable mappings. The name “normalizing flow” can be interpreted as the following:

1. “Normalizing” means that the change of variables gives a normalized density after applying an invertible transformation.
2. “Flow” means that the invertible transformations can be composed with each other to create more complex invertible transformations.

### Probability Density Distillation

### Additional Loss Terms

The KL loss alone does not constrain the distillation process enough to obtain natural sounding speech (e.g., low-volume audio suffices for the KL) so extra loss functions were used.

### Results

In all the experiments, models were conditioned on linguistic features (as in original WaveNet) and pitch information (log of fundamental frequency).  The sampling speed of the autoregressive model is 172 timesteps/second for whereas it is 500,000 timesteps/second for the distilled model showing a speed up of over 3 orders of magnitude.

![](/home/kritianandan/Pictures/result1.png)

Regarding audio fidelity, the MOS scores of autoregressive and distilled WaveNet were pretty much the same.

![](/home/kritianandan/Pictures/result2.png)

For multi-speaker generation, distilled WaveNet outperformed both baselines (parametric and concatenative).

![](/home/kritianandan/Pictures/result3.png)

To evaluate the improvement of the model over introducing the extra loss functions, the authors used A/B testing as the MOS scores of these models were very similar.

### References

[DeepMind](https://deepmind.com/blog/article/wavenet-launches-google-assistant)

[Normalizing Flows IEEE](https://arxiv.org/pdf/1908.09257.pdf)









