# 👗 Fashion Recommender System

Image-based fashion product recommender. Upload a photo of a fashion item and the app returns the most visually similar products from the catalogue.

Görsel tabanlı moda öneri sistemi: bir ürün fotoğrafı yükleyin, uygulama katalogdaki görsel olarak en benzer ürünleri önerir.

- 🤗 **Model (Hugging Face):** https://huggingface.co/tufanyldrm1989/fashion-recommender-model
- 🚀 **Live app (Streamlit):** _Streamlit uygulama linkinizi buraya yazın_
- 📓 **Training notebook:** `Fashion_Recommender_Model.ipynb` (in the Hugging Face repo)

## How it works / Nasıl çalışır

1. **Feature extraction:** A ResNet50 network pre-trained on ImageNet (without the classification head, followed by global max pooling) turns every product image into a 2048-dimensional vector. Vectors are L2-normalised.
2. **Similarity search:** `scikit-learn` `NearestNeighbors` (brute force, Euclidean distance) finds the closest vectors to the uploaded image.
3. **App:** A Streamlit interface shows the uploaded image and the top recommendations.

No extra training is needed; the pre-trained network is used as a feature extractor.

## Dataset

[Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) from Kaggle. This repository contains a 3,000-image subset so that it stays small enough to deploy.

## Project structure

```
├── app.py                  # Streamlit app
├── requirements.txt
├── images/                 # 3,000 catalogue images (subset)
├── Images_features.pkl     # image feature vectors (also on Hugging Face)
└── filenames.pkl           # image paths matching the feature vectors
```

The model files (`resnet50_feature_extractor.keras`, `Images_features.pkl`, `filenames.pkl`) are downloaded automatically from the Hugging Face repository if they are not present locally.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Python 3.11 is recommended for TensorFlow.

## Result

Using a pre-trained ResNet50 and a simple nearest-neighbour search, the system can find products that look similar to a given item. Model and features are saved and published on Hugging Face and used by the Streamlit app.
