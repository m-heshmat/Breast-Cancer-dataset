**Idea**


This project focuses on using various machine learning techniques
to classify and cluster breast cancer cases. The project employs a
wide array of classifiers methods, with a GUI, detailing their implementation
and functionality.

**Dataset**


The dataset for this project is sourced from a public breast cancer
dataset, including various attributes essential for cancer
classification and analysis.

**Preprocessing steps**


The preprocessing involves several key steps to optimize the
data:
1. Removing Unnecessary Columns: Drops columns like 'id'
which are not useful for the analysis.
2. Handling Missing Values: Ensures there are no missing
values in the dataset that could impair model performance.
3. Encoding Categorical Labels: Transforms categorical data
into numeric format using LabelEncoder.
4. Feature Scaling: Applies StandardScaler to standardize
features, ensuring equal importance is given to all features.
5. Outlier Detection and Removal: Employs the Interquartile
Range (IQR) method to detect and remove outliers,
enhancing model accuracy.

**Modeling techniques**

Classifiers:

1. Support Vector Machine (SVM): Implemented using SVC
from Scikit-learn, optimizing the hyperparameters for kernel
type and margin softness.
2. K-Nearest Neighbors (KNN): Uses KNeighborsClassifier,
with the number of neighbors as a variable to tune.
3. Decision Tree Classifier: Configured using
DecisionTreeClassifier, with adjustments for the depth of the
tree and criteria for splitting.
4. Gaussian Naive Bayes: Uses GaussianNB, suitable for
distributions in the dataset and requires no configuration.


**Libraries used**

1. Pandas
2. Scikit-learn
3. Seaborn
4. Matplotlib
5. tkinter
