# Model Weights

## Quick Start — Pre-trained Weights (Level 0)

To get the pipeline running immediately without training:

1. Go to [Kaggle](https://www.kaggle.com/) and search for "license plate detection YOLOv8"
2. Find a notebook/project with trained YOLOv8 weights
3. Download the model weights (usually named `best.pt`)
4. Place the file here as `models/best.pt`

Alternatively, search on [Roboflow Universe](https://universe.roboflow.com/) for
pre-trained license plate models.

## Fine-tuned Weights (Level 1)

After training on Google Colab using the Kaggle dataset
(see `notebooks/02_training.ipynb`):

1. Download `best.pt` from your Colab training run (saved to Google Drive)
2. Replace the file here as `models/best.pt`

## Dataset

We use the **Indian Number Plate Annotation** dataset from Kaggle:
https://www.kaggle.com/datasets/sunrajbishnolia/license-plate-detection

- 15,338 images (10,736 train / 3,068 val / 1,534 test)
- Native YOLO annotation format
- Single class: `license_plate`
