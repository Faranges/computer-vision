# Pneumonia Detection from Chest X-Rays using ResNet50 (CNN)

This project focuses on building, training, evaluating, and fine-tuning a Convolutional Neural Network (CNN) to detect pneumonia from chest X-ray images. The goal was high accuracy and medically meaningful performance, especially minimizing false negatives (missed pneumonia cases).   

The project uses transfer learning with ResNet50, followed by important fine-tuning steps. All the relevant evaluation metrics (accuracy, validation, confusion matrix, precicion, recall, f1-score) and visualization are included.

## Table of Contents
- [Workflow](#workflow)
- [Features](#features)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Evaluation](#model-evaluation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Conclusion](#conclusion)
- [Personal Note](#personal-note)
  
## Workflow
1. **Environment setup**
   
    * Created and activated a Python virtual environment
    * Installed required libraries (TensorFlow, NumPy, Matplotlib, Seaborn, scikit-learn, OpenCV)
2. **Dataset acquisition**
   
    * Downloaded the Chest X-Ray Pneumonia dataset from Kaggle
    * Reorganized the dataset into train / validation / test splits (80 / 10 / 10)
3. **Data preprocessing**
   
    Used ```ImageDataGenerator``` for loading and preprocessing images.  
    Applied ````preprocess_input```` specific to ResNet50
4. **Baseline model (Transfer Learning)**
   
    * Loading pretrained ResNet50 (ImageNet weights)
    * Freezing all convolutional layers
    * Adding a custom classification head
    * Training and evaluating baseline performance
5. **Fine-tuning**
   
   * Unfreeze top layers of ResNet50
   
    * Recompile with a lower learning rate
Continue training with EarlyStopping
6. **Model evaluation**
   
    * Accuracy & loss curves
    * Confusion matrix
    * Medical interpretation (recall, false negatives)
  
## Features
* Binary classification: Normal vs Pneumonia
* Transfer learning with ResNet50
* Fine-tuning of pretrained CNN layers
* Early stopping to prevent overfitting
* Strong focus on medical metrics (recall, false negatives)
* Clear visualizations for training and evaluation

## Dataset
**Source**
Kaggle: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/code/data) by Paul Mooney.

**Classes**    
* ````NORMAL````  
* ````PNEUMONIA````

**Structure after reorganization of the dataset:**
````
data/
└── chest_xray_reorganized/
    ├── train/
    │ ├── NORMAL/
    │ └── PNEUMONIA/
    ├── val/
    │ ├── NORMAL/
    │ └── PNEUMONIA/
    └── test/
    ├── NORMAL/
    └── PNEUMONIA/
````

The dataset is imbalanced (more pneumonia cases), which is taken into account during evaluation.

## Data Preprocessing
* Images are resized to **224×224** pixels (ResNet50 input size)
* RGB format is enforced
* Pixel values are normalized using:  
````preprocess_input````

This function applies the same normalization used during ResNet50’s original ImageNet training, ensuring compatibility with pretrained weights.

**No data augmentation** was applied, as aggressive transformations may distort medically relevant features in X-ray images.

## Model Evaluation
### Metrics used

* **Accuracy** – overall correctness
* **Recall (Sensitivity)** – especially for Pneumonia
* **Confusion Matrix** – detailed error analysis


### Visualizations included
#### 1. Training vs Validation Accuracy (Baseline)


#### 2. Training vs Validation Loss (Baseline)
#### 3. Training vs Validation Accuracy (Fine-tuned)
#### 4. Training vs Validation Loss (Fine-tuned)
#### 5. Confusion Matrix (Fine-tuned ResNet50)

### Key medical results (Fine-tuned model)
* Pneumonia recall ≈ 99%
* False negatives = 4 cases
* High sensitivity, suitable for screening support

EarlyStopping halted training at 6 epochs to prevent overfitting when validation loss stopped improving.

## Usage
### 1. Download the dataset from Kaggle
This project uses the Chest X-Ray Pneumonia dataset by Paul Mooney.
First, make sure you have the Kaggle CLI installed and configured.   

````pip install kaggle```` 

Place your ````kaggle.json```` API token in:

````~/.kaggle/kaggle.json````

Then download the dataset into the project directory:
````
cd path/to/your/project
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
````
Unzip the dataset: 

````unzip chest-xray-pneumonia.zip -d data/````

### 2. Reorganize the dataset
The original dataset structure is not ideal for Keras ````ImageDataGenerator````.

Run the provided reorganization script to create a clean train / val / test directory structure called data/chest_xray_reorganized:

````python reorganize_data.py````

### 3. Run the notebook
Activate your virtual environment and launch Jupyter:
````
source venv_computer_vision/bin/activate
jupyter notebook 
````
Open the main notebook and run all cells in order:
````notebooks/pneumonia_resnet50.ipynb````

## Project Structure
````computer-vision/
├── data/
│ └── chest_xray_reorganized/
│ ├── train/
│ ├── val/
│ └── test/
├── Pneumonia_Challenge.ipynb
├── venv_computer_vision/
├── .gitignore
├── README.md
└── requirements.txt
````

## Conclusion
This project demonstrates how transfer learning and fine-tuning can be effectively applied to medical imaging tasks. By prioritizing recall and minimizing false negatives, the final model achieves strong and medically meaningful performance while remaining interpretable and well-evaluated.

## Personal note
This project was done as part of the Data Science & AI Bootcamp at BeCode.org and took 4 days to complete.

## Future improvements
- ROC curve & AUC
- Grad-CAM visualization
- Model saving & inference on new images