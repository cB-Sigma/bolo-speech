# Models bolo does not run

Models that were looked at for the catalog and left out because no runtime
bolo has can run them yet. Every catalog entry is runnable; these are not in
the catalog. Each has the concrete reason and what would change it.

## Denoising

| model | licence | size | why it is not in the catalog |
| --- | --- | --- | --- |
| ClearVoice FRCRN_SE_16K ([alibabasglab/FRCRN_SE_16K](https://huggingface.co/alibabasglab/FRCRN_SE_16K)) | Apache-2.0 | 161 MB | PyTorch only; the network comes only with the `clearvoice` package (ClearerVoice-Studio), which pins numpy below 2, librosa 0.10.2, opencv, scenedetect and torchvision and needs `audioop-lts` on Python 3.13. A trial loader through that package ran about 50 times slower than real time on one shared 4-vCPU machine, and its output on a noisy sample correlated less with the clean speech than the noisy input did, so it was not added. No ONNX export is published. |
| ClearVoice MossFormer2_SE_48K ([alibabasglab/MossFormer2_SE_48K](https://huggingface.co/alibabasglab/MossFormer2_SE_48K)) | Apache-2.0 | 222 MB | the same package and dependencies as FRCRN; not tried after FRCRN's speed. |
| ClearVoice MossFormerGAN_SE_16K ([alibabasglab/MossFormerGAN_SE_16K](https://huggingface.co/alibabasglab/MossFormerGAN_SE_16K)) | Apache-2.0 | 39 MB (+ 98 MB discriminator, not needed) | the same package and dependencies as FRCRN; not tried after FRCRN's speed. |
| Resemble Enhance ([ResembleAI/resemble-enhance](https://huggingface.co/ResembleAI/resemble-enhance)) | MIT | 713 MB | the `resemble-enhance` package pins torch 2.1.1, deepspeed 0.12.4 and gradio 4.8; the enhancer is a DeepSpeed checkpoint with no ONNX export; over the 500 MB this project tries on its shared test machine. |
| DeepFilterNet 2 (`models/DeepFilterNet2_onnx.tar.gz` of [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) 0.5.6) | MIT or Apache-2.0 | 9 MB | the official ONNX export is three graphs (encoder, ERB decoder, deep-filter decoder) that need DeepFilterNet's own STFT, ERB features, normalisation and deep filtering around them (libDF); bolo has none of that. DeepFilterNet 3 is in the catalog (`deepfilternet3`) as torchDF's single streaming graph, which carries all of it. |
