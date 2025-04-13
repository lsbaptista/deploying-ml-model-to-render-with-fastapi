# Model Card

For additional information see the Model Card paper: [Model Card Paper](https://arxiv.org/pdf/1810.03993.pdf)

## Model Details

This model is a Logistic Regression classifier, created as part of a machine learning pipeline defined in the starter/starter project structure. The model uses the default hyperparameters from scikit-learn (>= 0.24). It was trained to predict whether an individual's income exceeds $50K based on demographic and employment features from the U.S. Census dataset. The project includes code for preprocessing the data, training, evaluating, and persisting the model.

## Intended Use

This model is intended to predict whether an individual's income exceeds $50K or not, based on features such as age, education, occupation, and other demographic and employment attributes. This model is useful for researchers, analysts, and policymakers who need to quickly classify individuals into income categories.

**Intended users**: Data scientists, analysts, or automated systems requiring binary income-level classification for individuals based on census data.

## Training Data

- **Source**: UCI Adult Census Income Dataset
- **Features**: age, workclass, fnlgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country
- **Label**: salary (<=50K or >50K)
- **Preprocessing**: 
  - Categorical features were encoded using OneHotEncoder.
  - The label (salary) was encoded using LabelBinarizer.
  - Continuous features were retained as-is.
- **Train/Test Split**: The dataset was split into 80% for training and 20% for testing using `train_test_split` from scikit-learn. K-Fold Cross Validation may be used for additional robustness.

## Evaluation Data

The evaluation data consists of 20% of the original dataset, which was separated using a train-test split. The test set contains a diverse distribution of demographic and employment features, allowing the model to be evaluated for its ability to generalize to unseen data.

## Metrics

The model was evaluated using the following metrics:

- **Precision**: Measures the proportion of positive predictions that were correct.
- **Recall**: Measures the proportion of actual positives that were correctly identified by the model.
- **F1 Score**: The harmonic mean of precision and recall, offering a balance between the two.

These metrics help to ensure that the model maintains a balance between identifying individuals with high income (positive class) and avoiding false positives or negatives.

## Ethical Considerations

This model may inherit biases from the data it was trained on, particularly with respect to:
- **Socioeconomic and racial biases**: The dataset contains inherent societal biases that may lead to skewed predictions for certain demographic groups.
- **Gender bias**: The data may reflect gender imbalances in certain occupations and income levels, potentially influencing the model's predictions.

It is recommended to use fairness audit tools such as Aequitas or Fairlearn to analyze and mitigate any disparities that might arise from these biases.

## Caveats and Recommendations

- **Data Quality**: The model's performance heavily depends on the quality and representativeness of the training data. If the data is outdated or unbalanced, the model's predictions may be less accurate.
- **Feature Engineering**: While continuous features were kept as-is, additional preprocessing or scaling could improve the model's performance, particularly if other machine learning models are used.
- **Bias**: As mentioned earlier, care should be taken to evaluate and address any potential biases, especially if the model is deployed in a real-world system where fairness is a concern.
- **Use Case**: This model is designed for binary income classification. It may not generalize well to other tasks, especially in different domains or datasets.


