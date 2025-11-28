# Forest Cover Type Classification (Machine Learning Project)

This project uses the **Forest Cover Type** dataset to build and evaluate machine learning models that predict the type of forest cover based on cartographic variables such as elevation, soil type, wilderness area, and geographic features. The workflow includes data loading, preprocessing, feature exploration, model training, evaluation, and interpretation of results.

The goal of the project is to understand which environmental features have the strongest influence on predicting forest cover type and to compare different machine learning approaches using scikit-learn.

---

## Features of the Project

### Data Preprocessing
- Load and clean the forest dataset  
- Handle missing or undefined values  
- Encode categorical variables such as soil type and cover type  
- Normalize or scale numerical features for model stability  

### Exploratory Analysis
- Identify the distribution of cover types  
- Review feature importance  
- Examine correlations between elevation, terrain, and soil indicators  

### Machine Learning Models
This project implements and compares multiple classification models including:

- Random Forest Classifier  
- Decision Tree Classifier  
- Logistic Regression  
- Support Vector Machine (optional)  

### Model Evaluation
- Split data into training and testing sets  
- Evaluate using accuracy, confusion matrix, and classification report  
- Extract feature importances for interpretability  
- Analyze which variables contribute most to correct predictions  

---

## Project Structure

```
forest-ml/
│
├── data/
│   └── forest.csv                # Forest cover dataset
│
├── forest_analysis.ipynb         # Notebook with preprocessing, training, and results
├── forest_ml.py                  # Python script version (optional)
│
├── requirements.txt              # Dependencies (scikit-learn, pandas, numpy, matplotlib)
└── README.md
```

---

## Installation

Create a virtual environment (optional but recommended):

```
python3 -m venv venv
source venv/bin/activate     # macOS / Linux
venv\Scripts\activate        # Windows
```

Install dependencies:

```
pip install -r requirements.txt
```

If running the notebook:

```
pip install jupyterlab
jupyter lab
```

---

## Running the Notebook

Open:

```
forest_analysis.ipynb
```

Inside the notebook you can:

- Explore the dataset  
- Train models  
- View accuracy metrics  
- Visualize feature importance  

---

## Example Requirements File

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---

## Example Output (Random Forest)

- Accuracy: ~75–85 percent depending on hyperparameters  
- Top features often include:
  - Elevation  
  - Hillshade at 3 pm  
  - Vertical distance to hydrology  
  - Soil type indicators  

These patterns align with physical geography expectations: elevation and soil composition play major roles in determining forest cover type.

---

## Summary

This project demonstrates a complete workflow for solving a multi-class classification problem using real environmental data. It highlights how preprocessing, feature engineering, and model selection contribute to performance, and it provides a clear, interpretable analysis of which factors influence forest cover predictions.

