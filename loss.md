# Audio Watermarking Loss Functions

## 1. L1 Loss on the Delta

**What it is:**  
The L1 loss measures the absolute difference between the generated watermark perturbation (delta) and a zero tensor.

**Function & Use:**
```python
loss_l1 = F.l1_loss(delta, torch.zeros_like(delta))
```
This loss penalizes large deviations in the generated delta. By pushing delta toward zero, it forces the generator to make only minimal modifications to the original audio.

**Why it's important:**  
Keeping the delta small is crucial for maintaining the imperceptibility of the watermark. Large perturbations could make the watermark audible or degrade the quality of the original signal.

## 2. MultiScaleMelLoss (or SimpleMelLoss)

**What it is:**  
This loss compares the mel spectrograms of the original and watermarked audio. In your code, you have implemented a version (often called SimpleMelLoss) that computes the difference between log-mel spectrograms.

**Function & Use:**
```python
mel_orig = torch.log(self.mel_spec(original) + 1e-5)
mel_wm   = torch.log(self.mel_spec(watermarked) + 1e-5)
loss = F.l1_loss(mel_orig, mel_wm)
```
It calculates the L1 loss between these log-mel spectrograms.

**Why it's important:**  
The mel spectrogram captures perceptually relevant frequency content. By minimizing the difference between the original and watermarked mel spectrograms, this loss ensures that the spectral (and thus perceptual) quality of the audio is preserved despite the watermark embedding.

## 3. TFLoudnessLoss

**What it is:**  
The Time–Frequency (TF) Loudness Loss compares the STFT representations of the original and watermarked audio over several frequency bands.

**Function & Use:**  
This loss divides the frequency spectrum into several bands and, for each band, computes:

- **Loudness Difference:** An L1 loss on the logarithm of the band energies.
- **Spectral Shape Difference:** An MSE loss between the magnitudes.
- **Phase Difference:** A measure using the cosine of the phase difference.

Together, the loss is computed as:
```python
return loudness_loss + spectral_loss + 0.2 * phase_loss
```

**Why it's important:**  
This loss ensures that both the energy distribution (loudness) and spectral structure remain similar after watermarking. It also takes phase differences into account (albeit with a lower weight) so that the temporal characteristics are preserved. This makes the watermark imperceptible in terms of both time and frequency.

## 4. Adversarial Loss

**What it is:**  
The adversarial loss is derived from a discriminator network trained to differentiate between original and watermarked audio.

**Function & Use:**  
The discriminator is trained to output:
- 1 for original audio and
- 0 for watermarked audio.

The generator, on the other hand, is penalized (via a binary cross-entropy loss) if the discriminator can correctly identify the watermarked audio.

```python
gen_loss = F.binary_cross_entropy_with_logits(fake_output, torch.ones_like(fake_output))
```
(with the discriminator being trained separately within the same function).

**Why it's important:**  
This loss forces the generator to produce watermarked audio that is indistinguishable from the original to an adversary. By "tricking" the discriminator, the generator learns to embed the watermark in a way that does not introduce perceptible artifacts.

## 5. Masked Localization Loss

**What it is:**  
This loss focuses on the output of the detector network's first channel—the detection probability per time step—and compares it with a binary mask indicating where the watermark is present.

**Function & Use:**  
It applies label smoothing and focal loss weighting to emphasize "hard" examples:

```python
det_prob = detector_out[:, 0:1, :]
smoothed_mask = mask * (1.0 - smooth_eps) + (1.0 - mask) * smooth_eps
focal_weight = (1 - pt) ** 2
loss = (focal_weight * F.binary_cross_entropy(det_prob, smoothed_mask, reduction='none')).mean()
```

**Why it's important:**  
This loss trains the detector to accurately localize the watermark. In scenarios where only parts of the audio are watermarked, this becomes crucial for robust detection.

## 6. Decoding Loss

**What it is:**  
The decoding loss measures how well the embedded watermark message (a set of bits) can be recovered from the detector's output.

**Function & Use:**  
It first extracts the bit probability maps from the detector (channels beyond the first) and averages them over time. Then it converts the true message (an integer) into a binary vector and computes a focal-weighted binary cross-entropy loss between the predicted bit probabilities and the true bits:

```python
# Convert integer message to bits and compute binary cross-entropy loss with focal weighting:
for i in range(b):
    bit_i = ((message >> i) & 1).float()
msg_bits = torch.stack(msg_bits, dim=1)
focal_weight = (1 - pt) ** gamma
loss = (focal_weight * F.binary_cross_entropy(bit_prob, msg_bits, reduction='none')).mean()
```

**Why it's important:**  
Even if the watermark is imperceptible, it must carry useful information. The decoding loss ensures that the embedded message is robustly recoverable, which is vital for applications like copyright protection or covert communications.

## How They Are Used Together

In your training loop, these losses are computed on each batch and then combined in a weighted sum (using the lambda weights defined in your hyperparameters). This composite loss drives the training of both the generator and detector:

```python
loss = (lambda_L1     * loss_l1 +
        lambda_msspec * loss_msspec +
        lambda_adv    * loss_adv +
        lambda_loud   * loss_loud +
        lambda_loc    * loss_loc +
        lambda_dec    * loss_dec)
```

Each loss plays a role:

- **L1 and Mel Losses** ensure the watermarked audio remains perceptually similar to the original.
- **TF-Loudness Loss** enforces similarity in time–frequency structure.
- **Adversarial Loss** pushes the generator to create imperceptible watermarks.
- **Localization Loss** trains the detector to find where the watermark is.
- **Decoding Loss** ensures the watermark message can be accurately recovered.