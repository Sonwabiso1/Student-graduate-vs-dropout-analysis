# Student Dropout and Graduation Prediction

## Project Overview
This project aims to analyze and predict student outcomes, specifically focusing on whether a student will **graduate** or **drop out** based on various academic, socioeconomic, and personal factors. The primary goal is to identify key predictors of student success and to develop a model that can predict these outcomes using historical data.

### Key Objectives:
- Perform data cleaning and preprocessing.
- Explore correlations between features (grades, parental education, economic factors, etc.) and student outcomes.
- Perform hypothesis testing to validate relationships between key factors and student success.
- Build predictive models to classify students as likely to **graduate** or **drop out**.

---

## Dataset

The dataset contains various student features and a target variable indicating whether they graduated or dropped out.

### Key Features:
- **Marital status**: Indicates the student's marital status.
- **Application mode**: The type of application mode used.
- **Previous qualification (grade)**: The grade obtained in a previous qualification.
- **Curricular units 1st sem (grade)**: Student's performance in the first semester.
- **Curricular units 2nd sem (grade)**: Student's performance in the second semester.
- **Parental qualifications**: Levels of education for the student's parents.
- **Unemployment rate**, **Inflation rate**, and **GDP**: Socioeconomic factors that might influence academic success.
- **Target**: The outcome, where `1` indicates **Graduate** and `0` indicates **Dropout**.

### Target Variable:
- **1** = Graduate
- **0** = Dropout

---

## Installation

### Requirements:
- Python 3.x
- Required libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scipy`
  - `seaborn` (optional, for more advanced visualizations)

You can install the required libraries using pip:
```bash
pip install pandas numpy matplotlib scipy seaborn
```

### Cloning the Repository
Clone the repository to your local machine using:
```bash
git clone https://github.com/Sonwabiso1/Student-graduate-vs-dropout-analysis.git
cd Student-graduate-vs-dropout-analysis
```

---

## Usage

1. **Preprocess the Data**:
   - Load the dataset, clean any missing or incorrect data, and preprocess the features.
   
   Example:
   ```python
   import pandas as pd
   data = pd.read_csv('Predict_Student_Dropout_and_Academic_Success.csv', delimiter=';')
   data['Target'] = data['Target'].map({'Graduate': 1, 'Dropout': 0})
   ```

2. **Run Exploratory Data Analysis**:
   - Explore the data to understand the distributions, relationships, and outliers.
   - Calculate correlations between the target variable and other features.
   
   Example:
   ```python
   correlation_matrix = data.corr()
   correlation_matrix[['Target']].sort_values(by='Target', ascending=False)
   ```

3. **Hypothesis Testing**:
   - Perform statistical tests like t-tests to check the significance of relationships between student grades and their outcomes.
   
   Example:
   ```python
   from scipy import stats
   grads = data[data['Target'] == 1]
   drops = data[data['Target'] == 0]
   
   t_test_1st_sem = stats.ttest_ind(grads['Curricular units 1st sem (grade)'], drops['Curricular units 1st sem (grade)'], equal_var=False)
   print('T-test for 1st semester grades:', t_test_1st_sem)
   ```

4. **Visualizations**:
   - Visualize the relationships using histograms, boxplots, and other plots to better understand the data.
   
   Example:
   ```python
   import matplotlib.pyplot as plt
   plt.boxplot([grads['Curricular units 2nd sem (grade)'], drops['Curricular units 2nd sem (grade)']], labels=['Graduates', 'Dropouts'])
   plt.title('2nd Semester Grades Comparison')
   plt.show()
   ```

5. **Modeling** (optional):
   - Build a predictive model to classify students as **Graduate** or **Dropout** using machine learning models like Logistic Regression, Decision Trees, etc.

---

## Analysis Insights

### Key Findings:
- **Grades**: Students with higher grades in the first and second semesters are more likely to graduate.
- **Parental Influence**: There is little to no correlation between parental education levels and student graduation rates.
- **Socioeconomic Factors**: Unemployment, inflation, and GDP have minimal impact on whether students graduate or drop out.

### Hypotheses Tested:
- **Grade Hypothesis**: Students with higher grades are more likely to graduate, which was confirmed through correlation analysis and t-tests.
- **Parental Education Hypothesis**: Parental education levels do not significantly impact student success, confirmed through hypothesis testing.

---

## Future Work
- Improve predictive modeling accuracy by incorporating additional features or experimenting with more complex machine learning models.
- Explore time-series analysis if additional data over time becomes available.

---

## Contributing

If you'd like to contribute to this project, feel free to submit a pull request or open an issue for discussion.

---

## Acknowledgements
Special thanks to the data providers and educational institutions that made this analysis possible.
This data was obtained from this source: [kaggle](https://www.kaggle.com/datasets/syedfaizanalii/predict-students-dropout-and-academic-success).

