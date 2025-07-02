Crop Type Classification with Satellite Imagery

This project involves classifying crop types using multi-spectral satellite images provided by Radiant Earth. It includes data download, preprocessing, feature extraction, and model training to predict crop types.

Project Structure

Data Download: Uses Radiant Earth API to download satellite imagery and field labels.

Feature Extraction:

Image pixel values

Spectral indices (like NDVI, AVI)

Spatial features (e.g., area of field)

Modeling Approaches:

Approach 1: Uses all features (pixels + indices + spatial features).

Approach 2: Uses only pixel values and statistics.

Both approaches use:

CatBoost Classifier (with and without class weights)

Bagged Linear Discriminant Analysis (LDA)

Ensembling (weighted average) is used to combine the models.

How It Works

Download satellite imagery and labels using the API.

Extract relevant pixel and statistical features for each field.

Train models using the extracted features.

Predict crop type for each field.

Generate a CSV submission file.

Requirements

sklearn==0.22.2
eli5==0.10.1
catboost==0.22
scipy==1.4.1
numpy==1.18.3
pandas==1.0.3
tifffile

