# Crop Classification Using Remote Sensing and Machine Learning

## Abstract
This project investigates advanced crop classification using multi-temporal satellite imagery and machine learning. We explore several feature engineering techniques, ensemble models, and spatial analysis to improve classification accuracy on the [Radiant MLHub African Crops Dataset](https://mlhub.earth/data/ref_african_crops_kenya_02/).

## Motivation & Background
Crop classification is a fundamental task in precision agriculture and food security monitoring. Accurate crop mapping enables better resource allocation, yield estimation, and policy making. This project explores state-of-the-art approaches to multi-class crop identification using Sentinel-2 imagery and open-source tools.

## Dataset
- **Source:** Radiant MLHub African Crops Kenya 02
- **Type:** Multi-temporal Sentinel-2 imagery and field-level labels
- **Features:** Pixel values, vegetation indices (NDVI, AVI, etc.), spatial features

## Methodology
- **Data Acquisition:** Automated download and organization of satellite imagery and labels.
- **Preprocessing:** Image cleaning, feature extraction (spectral indices, spatial features).
- **Exploratory Data Analysis:** Visualizations, class distributions, and correlation analysis.
- **Modeling:** Ensemble of CatBoost, Random Forest, and LDA with class balancing.
- **Evaluation:** Log-loss, per-pixel and per-field metrics, cross-validation.

## Experiments & Results
- Multiple feature sets compared (raw pixels, indices, spatial).
- Ensemble models outperform individual learners.
- Detailed results and figures in `notebooks/` and `reports/`.

## Discussion & Future Work
- Investigate deep learning (CNNs) for further improvement.
- Integrate temporal analysis for crop growth stage modeling.
- Explore transfer learning and domain adaptation.

## How to Reproduce
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Download data using provided scripts
4. Run notebooks in order

## References
See `references/references.bib` for citations and key literature.

---

## Author
**B. Sai Varshith Reddy**  
Researcher in Machine Learning & Remote Sensing  
Contact: saivarshithreddybonala@gmail.com  

This project is a research effort for advanced crop classification and is intended for academic and educational use.
