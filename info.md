Input: Audio waveform (1, 16000 samples)
         │
         ▼
 ┌─────────────────────────────┐
 │  Conv1d: 1 → 16             │
 │  kernel=15, stride=1, pad=7   │
 └─────────────────────────────┘
         │
         ▼
       ReLU
         │
         ▼
 ┌─────────────────────────────┐
 │  Conv1d: 16 → 32            │
 │  kernel=15, stride=2, pad=7   │
 └─────────────────────────────┘
         │
         ▼
       ReLU
         │
         ▼
 ┌─────────────────────────────┐
 │  Conv1d: 32 → 64            │
 │  kernel=15, stride=2, pad=7   │
 └─────────────────────────────┘
         │
         ▼
       ReLU
         │
         ▼
   (Output shape: [batch, 64, 4000])
         │
         ▼
    Transpose to [batch, 4000, 64]
         │
         ▼
 ┌─────────────────────────────┐
 │       LSTM Layer            │
 │   (1 layer, hidden=64)      │
 └─────────────────────────────┘
         │
         ▼
    (Output shape remains [batch, 4000, 64])
         │
         ▼
   Transpose back to [batch, 64, 4000]
         │
         ▼
 ┌─────────────────────────────┐
 │ ConvTranspose1d: 64 → 32    │
 │ kernel=15, stride=2, pad=7,   │
 │   output_padding=1          │
 └─────────────────────────────┘
         │
         ▼
       ReLU
         │
         ▼
 ┌─────────────────────────────┐
 │ ConvTranspose1d: 32 → 16    │
 │ kernel=15, stride=2, pad=7,   │
 │   output_padding=1          │
 └─────────────────────────────┘
         │
         ▼
       ReLU
         │
         ▼
 ┌─────────────────────────────┐
 │   Conv1d: 16 → 1            │
 │  kernel=15, stride=1, pad=7   │
 └─────────────────────────────┘
         │
         ▼
   Apply tanh then scale by 0.01
         │
         ▼
Output: Watermark Delta (added to original audio)
         (Same shape as input: [1, 16000])



Input: Watermarked audio waveform (1, 16000 samples)
         │
         ▼
 ┌─────────────────────────────┐
 │  Conv1d: 1 → 16             │
 │  kernel=15, stride=1, pad=7   │
 └─────────────────────────────┘
         │
         ▼
       ReLU
         │
         ▼
 ┌─────────────────────────────┐
 │  Conv1d: 16 → 32            │
 │  kernel=15, stride=2, pad=7   │
 └─────────────────────────────┘
         │
         ▼
       ReLU
         │
         ▼
 ┌─────────────────────────────┐
 │  Conv1d: 32 → 64            │
 │  kernel=15, stride=2, pad=7   │
 └─────────────────────────────┘
         │
         ▼
       ReLU
         │
         ▼
 ┌─────────────────────────────┐
 │ ConvTranspose1d: 64 → 32    │
 │ kernel=15, stride=2, pad=7,   │
 │   output_padding=1          │
 └─────────────────────────────┘
         │
         ▼
       ReLU
         │
         ▼
 ┌─────────────────────────────┐
 │ ConvTranspose1d: 32 → 16    │
 │ kernel=15, stride=2, pad=7,   │
 │   output_padding=1          │
 └─────────────────────────────┘
         │
         ▼
       ReLU
         │
         ▼
 ┌─────────────────────────────┐
 │   Conv1d: 16 → 1            │
 │  kernel=15, stride=1, pad=7   │
 └─────────────────────────────┘
         │
         ▼
   Apply sigmoid activation
         │
         ▼
Output: Watermark Detection Map (probabilities for each sample,
         shape: [1, 16000])
