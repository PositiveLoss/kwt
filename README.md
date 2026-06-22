# Torch-KWT
Unofficial PyTorch implementation of [*Keyword Transformer: A Self-Attention Model for Keyword Spotting*](https://arxiv.org/abs/2104.00769).

<a href="https://colab.research.google.com/github/ID56/Torch-KWT/blob/main/notebooks/Torch_KWT_Tutorial.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>


## Setup

```
uv sync
```

## Dataset
To download the Google Speech Commands V2 dataset, you may run the provided bash script as below. This would download and extract the dataset to the "destination path" provided.

```
./download_gspeech_v2.sh <destination_path>
```

## Training

The Speech Commands V2 dataset provides a "validation_list.txt" file and a "testing_list.txt" file. `train.py` will generate `training_list.txt`, `validation_list.txt`, `testing_list.txt`, and `label_map.json` automatically if they are missing. You can also create them manually with:

```
python make_data_list.py -v <path/to/validation_list.txt> -t <path/to/testing_list.txt> -d <path/to/dataset/root> -o <output dir>
```

This will create the files `training_list.txt`, `validation_list.txt`, `testing_list.txt`, and `label_map.json` at the specified output dir.

Running `train.py` is fairly straightforward. Only a path to a config file is required. Inside the config file, you'll need to add the paths to the .txt files and the label_map.json file created above.

```
python train.py --conf path/to/config.yaml
```

Training writes safetensors checkpoints to the run directory as `best.safetensors` and `last.safetensors`.

For cochleagram features, use the provided alternate config:

```
python train.py --conf config_cochleagram.yaml
```

For CoNNear cochlea features, download the pretrained CoNNear weights once and use the provided alternate config:

```
uv run python download_connear_model.py
python train.py --conf config_connear.yaml
```

Set `exp.warm_cache: True` to iterate the train, validation, and test dataloaders once before they are used. This warms preprocessing, worker, and OS file caches inside the training process.

Set `exp.feature_cache: True` to persist deterministic feature extraction under `exp.feature_cache_dir` and reuse it across runs. Feature caching skips waveform-level augmentation because it loads features directly; `spec_aug` still runs on cached features during training.

Cepstral and cochleagram extraction use [spafe-rs](https://github.com/RustedBytes/spafe). Set `hparams.audio.feature_type` to `mfcc`, `pncc`, `gfcc`, `ngcc`, `bfcc`, `rplp`, `cochleagram`, or `connear`. `config_connear.yaml` uses all 201 pretrained CoNNear channels with a `201 x 98` model input. CoNNear uses the pretrained PyTorch model from [PositiveLoss/CoNNear_cochlea-PyTorch](https://github.com/PositiveLoss/CoNNear_cochlea-PyTorch).

CoNNear is a neural feature extractor, so its first cache build is much slower than MFCC. `config_connear.yaml` enables disk feature caching by default and batches CoNNear extraction with `hparams.audio.connear_batch_size`.

Set `hparams.grad_accum_steps` above `1` to accumulate gradients across multiple dataloader batches before each optimizer update. The effective batch size is `batch_size * grad_accum_steps`.

The default optimizer/scheduler is `adamw_fused` with `one_cycle_lr`, which uses fused AdamW on CUDA when available and typically reaches useful learning rates faster than a long external warmup. Use `opt_type: adamw` and `scheduler_type: cosine_annealing` if you want the original-style baseline.

Resume interrupted training from the latest safetensors checkpoint:

```
python train.py --conf config.yaml --resume runs/exp-0.0.1/last.safetensors
```

Refer to the [example config](config.yaml) to see how the config file looks like, and see the [config explanation](docs/config_file_explained.md) for a complete rundown of the various config parameters.

## Inference

You can use the pre-trained model (or a model you trained) for inference, using the two scripts:

- `inference.py`: For short ~1s clips, like the audios in the Speech Commands dataset
- `window_inference.py`: For running inference on longer audio clips, where multiple keywords may be present. Runs inference on the audio in a sliding window manner.

```
python inference.py --conf config.yaml \
                    --ckpt <path to model.safetensors> \
                    --inp <path to audio.wav / path to audio folder> \
                    --out <output directory> \
                    --lmap label_map.json \
                    --device auto \
                    --batch_size 8   # should be possible to use much larger batches if necessary, like 128, 256, 512 etc.

python window_inference.py --conf config.yaml \
                    --ckpt <path to model.safetensors> \
                    --inp <path to audio.wav / path to audio folder> \
                    --out <output directory> \
                    --lmap label_map.json \
                    --device auto \
                    --wlen 1 \
                    --stride 0.5 \
                    --thresh 0.85 \
                    --mode multi
```
For detailed usage example, check the colab tutorial.

## ONNX Export

```
python export_onnx.py --conf config.yaml \
                      --ckpt <path to model.safetensors> \
                      --out exports/kwt.onnx \
                      --slim \
                      --verify
```

## Helion Kernels

`use_helion_kernels: True` enables optional Helion GELU forward/backward kernels in the transformer MLP. Helion currently targets Linux GPU environments with Triton; CPU and MPS runs fall back to PyTorch GELU.

## Tutorials
- [Colab Tutorial: [Using pretrained model | Inference scripts | Training]](https://colab.research.google.com/github/ID56/Torch-KWT/blob/main/notebooks/Torch_KWT_Tutorial.ipynb)

## Trackio

You can also set `exp.trackio: True` in `config.yaml` to log the same training metrics with [Trackio](https://github.com/gradio-app/trackio). By default Trackio logs locally; set `trackio_space_id` or `trackio_server_url` when you want to send runs to a remote dashboard.

## Pretrained Checkpoints

New training runs save safetensors checkpoints. The original KWT-1 pretrained file below is a legacy PyTorch checkpoint and remains loadable for compatibility.

| Model Name | Test Accuracy | Link |
| ---------- | ------------- | ---- |
|    KWT-1   |     95.98*     | [kwt1-v01.pth](https://drive.google.com/uc?id=1y91PsZrnBXlmVmcDi26lDnpl4PoC5tXi&export=download) |

*The [example config file](config.yaml) provided contains the exact settings used to train the KWT-1 checkpoint, and should be reproducible.
