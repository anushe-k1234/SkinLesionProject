# SkinLesionProject
Project to identify types of skin lesions

## Project Summary

• Developed a Python-based AI model using PyTorch to classify skin lesion images into seven diagnostic categories with a ResNet18 architecture.

• Leveraged libraries such as PyTorch, torchvision, pandas, scikit-learn, matplotlib, and seaborn to load data, preprocess images, train the model, and analyze results.

• Used transfer learning by fine-tuning a pretrained ResNet18 model with customized final layers for 7 skin lesion classes.

• Evaluated model performance using metrics including accuracy, precision, recall, and F1 score, supported by confusion matrices and ROC curves.

• Applied Grad-CAM visualizations to interpret model predictions by highlighting important regions in the skin lesion images.

## Project Structure

- 01_data_preparation.ipynb  
  Prepares data and image resizing for modeling.

- 02_eda_visualization.ipynb  
  Explores class distributions and shows sample images and augmentations.

- 03_model_training.ipynb  
  Defines and trains the ResNet18 model using transfer learning.

- 04_results_figures_report.ipynb  
  Creates evaluation plots such as confusion matrix, ROC curves, and Grad-CAM heatmaps.

- 05_evaluation_export_outputs.ipynb  
  Saves predictions, confidence scores, metrics, and ROC data as CSV files for further analysis or reporting.

## Dataset

Using the ISIC 2018 Challenge Task 3 skin lesion dataset with 7 different classes:

- MEL (Melanoma)  
- NV (Melanocytic nevus)  
- BCC (Basal cell carcinoma)  
- AKIEC (Actinic keratosis)  
- BKL (Benign keratosis)  
- DF (Dermatofibroma)  
- VASC (Vascular lesion)  

Images are resized to 224x224 pixels for input to the model.

## Model Details

- Architecture: ResNet18 pretrained on ImageNet  
- Final layer modified for 7 classes  
- Loss: CrossEntropyLoss with class weights  
- Optimizer: Adam  
- Learning rate scheduler: StepLR reducing LR every 3 epochs  
- Batch size: 16  
- Epochs: 6

## Outputs

Files generated include:

- Predictions CSV with true and predicted labels plus class probabilities  
- Metrics summary CSV (accuracy, precision, recall, F1)  
- Class distribution CSV  
- ROC curve data CSV for all classes  
- Grad-CAM metadata CSV linking images to heatmaps  

## Key Findings

- Highest accuracy on NV class due to large sample size  
- Challenges with rare classes like DF and VASC due to class imbalance  
- Grad-CAM visualizations show the model focuses on meaningful image regions for diagnosis  
