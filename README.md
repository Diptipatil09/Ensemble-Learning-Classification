# Ensemble Learning Techniques – Classification

##  Project Overview

This project implements and compares three popular Ensemble Learning techniques: **Stacking, AdaBoost, and Voting Classifier** using Python and Scikit-learn.

The objective is to combine multiple machine learning classifiers to improve classification performance and compare their results with individual base classifiers using different evaluation metrics.

The experiment uses the **Breast Cancer Wisconsin Dataset** to perform binary classification and evaluate the performance of different machine learning models.

##  Objectives

- Implement a Stacking Ensemble using multiple base classifiers and a meta-classifier.
- Implement an AdaBoost Ensemble using Decision Tree Stumps as weak learners.
- Implement a Voting Ensemble using majority voting.
- Compare ensemble models with individual machine learning classifiers.
- Evaluate model performance using Accuracy, Precision, Recall, and F1-score.
- Visualize model performance using graphs and confusion matrices.

##  Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| NumPy | Numerical operations |
| Pandas | Data handling |
| Scikit-learn | Machine learning algorithms |
| Matplotlib | Data visualization |
| Seaborn | Confusion matrix visualization |
| VS Code | Development environment |

##  Dataset

**Dataset Name:** Breast Cancer Wisconsin Dataset

The dataset is available directly through the Scikit-learn library.

Dataset details:
- Total Samples: 569
- Input Features: 30
- Classification Type: Binary Classification
- Target Classes: Benign and Malignant

The dataset is divided into:
- Training Data: 80% (455 samples)
- Testing Data: 20% (114 samples)

##  Machine Learning Models

### 1. Individual Base Classifiers

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes

### 2. Ensemble Learning Techniques

**Stacking Classifier**
- Combines multiple base classifiers.
- Uses Logistic Regression as the meta-classifier.
- Uses 5-fold cross-validation.

**AdaBoost Classifier**
- Uses Decision Tree Stumps as weak learners.
- Number of estimators: 100.
- Learning rate: 0.5.
- Focuses on incorrectly classified samples during training.

**Voting Classifier**
- Combines predictions from multiple classifiers.
- Uses Hard Voting (Majority Voting).
- Selects the class receiving the maximum number of votes.

##  Evaluation Metrics

The models are evaluated using the following classification metrics:

1. **Accuracy:** Measures the overall percentage of correct predictions.
2. **Precision:** Measures the correctness of positive predictions.
3. **Recall:** Measures the ability to identify actual positive samples.
4. **F1-score:** Measures the balance between precision and recall.

##  Project Structure

```text
Ensemble-Learning/
│
├── ensemble_learning.py
├── README.md
├── requirements.txt
│
├── performance_comparison.png
└── confusion_matrices.png
```

##  Installation and Execution

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Navigate to the Project Folder

```bash
cd Ensemble-Learning
```

### Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 4: Run the Python Program

```bash
python ensemble_learning.py
```

### Step 5: View the Results

After successful execution, the program displays:
- Individual classifier performance.
- Stacking Ensemble results.
- AdaBoost Ensemble results.
- Voting Ensemble results.
- Performance comparison table.
- Best-performing model.

It also generates two visualization files:

- `performance_comparison.png`
- `confusion_matrices.png`

##  Results and Observations

The experiment successfully implemented and evaluated seven machine learning models.

Based on the generated performance comparison graph:

- Logistic Regression achieved approximately 98% performance.
- Stacking achieved approximately 97% performance.
- Voting achieved approximately 97% performance.
- KNN and AdaBoost achieved approximately 96% performance.
- Naive Bayes achieved approximately 93% performance.
- Decision Tree achieved approximately 92% performance.

Logistic Regression showed the highest performance in this experiment, while Stacking and Voting demonstrated strong ensemble classification performance.

*Note: These percentages are visual estimates from the generated graph. Refer to the actual program output for exact metric values.*

##  Future Scope

- Implement advanced ensemble algorithms such as Random Forest and Gradient Boosting.
- Experiment with XGBoost and other boosting techniques.
- Apply hyperparameter tuning to improve model performance.
- Explore weighted voting techniques.
- Test the models on different datasets.
- Perform cross-validation for more reliable performance evaluation.

##  Conclusion

This project demonstrates the implementation and comparison of Stacking, AdaBoost, and Voting ensemble learning techniques using Python and Scikit-learn.

The experiment provides practical knowledge of combining multiple machine learning classifiers and evaluating their performance using classification metrics.

Ensemble learning can improve prediction reliability and is useful for solving various real-world classification problems.

##  Author

**Dipti Dipak Patil**

TY B.Tech. – Computer Engineering  
Artificial Intelligence and Machine Learning

---

**Project:** Ensemble Learning Techniques – Classification  
**Language:** Python  
**Platform:** Visual Studio Code  
