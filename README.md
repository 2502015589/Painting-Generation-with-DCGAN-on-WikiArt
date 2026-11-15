# Painting Generation with DCGAN on WikiArt

PyTorch DCGAN for generating 64×64 paintings from WikiArt, with FID and Inception Score evaluation.

![Fixed-noise samples at epoch 100](examples/epoch_100.png)

## Run the notebook

Use Python 3.11–3.13 and launch JupyterLab from this repository's root.

```sh
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` in Bash. For an NVIDIA GPU, install a CUDA-enabled PyTorch 2.6.0 / torchvision 0.21.0 build suitable for your driver before installing the remaining dependencies.

```sh
python -m pip install -r requirements.txt
python -m jupyterlab
```

Open `as2.ipynb`, select the environment's Python kernel, and run the cells in order. The notebook defaults to 100 epochs. For a short smoke run, lower `EPOCHS` and `EVAL_IMAGES`.

Place the original `wikiart-train-*.arrow` shards directly in `dataset/` if you already have them. Otherwise, the notebook downloads the pinned [HugGAN WikiArt dataset](https://huggingface.co/datasets/huggan/wikiart) on its first run. Allow roughly 65 GiB or more for the download, processed data, and resized cache. The notebook reuses the saved 64×64 cache on subsequent runs. Evaluation downloads pretrained Inception weights once.

Training writes samples, losses, metrics, and checkpoints to `dcgan_results/`. To continue a run, set `RESUME = True` and set `EPOCHS` to the desired total. Use only checkpoints you created or trust. The repository excludes data and checkpoints.

## Experiment

The unconditional DCGAN uses 100-dimensional Gaussian noise, 64 base channels in each network, batch size 64, and Adam with learning rate 0.0002 and betas (0.5, 0.999). The split uses seed 42, with 73,299 training images and 8,145 held-out images.

| Epoch | FID ↓ | IS ↑ |
| ---: | ---: | ---: |
| 3 | 242.15 | 2.50 |
| 20 | 69.06 | 5.05 |
| 30 | 58.91 | 4.92 |
| 50 | **51.76** | 5.14 |
| 70 | 53.23 | **5.27** |
| 100 | 51.88 | 5.18 |

Following the [PyTorch-Ignite evaluation tutorial](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/), IS uses 5,000 generated images. FID compares those images with 5,000 held-out WikiArt images using 2,048-dimensional pytorch-fid features.

See the [training loss curves](examples/loss_curves.png) and [epoch-100 metrics](examples/metrics.json). The model follows the [DCGAN architecture](https://arxiv.org/abs/1511.06434).

