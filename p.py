import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('breast-cancer.csv')

def DataCleaning():
    """
    Function to clean the dataset:
    - Drop 'id' column
    - Encode 'diagnosis' column
    - Standardize feature columns
    - Remove outliers based on IQR
    """
    df = pd.read_csv('breast-cancer.csv')
    df.drop('id', axis=1, inplace=True)
    
    # Encode 'diagnosis' column
    label_encoder = LabelEncoder()
    df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])
    
    # Standardize features
    features = df.columns.drop('diagnosis')
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # Remove outliers using Z-score
    z_scores = (df[features] - df[features].mean()) / df[features].std()
    outlier_condition = (z_scores.abs() < 3).all(axis=1)
    df_cleaned = df[outlier_condition]
    
    return df_cleaned

# Clean dataset
df_cleaned = DataCleaning()

# Split data into features (X) and target (y)
X = df_cleaned.drop('diagnosis', axis=1)
y = df_cleaned['diagnosis']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def update_text_area(text):
    """
    Function to update the text area in the GUI with provided text.
    """
    text_area.config(state=tk.NORMAL)
    text_area.delete('1.0', tk.END)
    text_area.insert(tk.END, text)
    text_area.config(state=tk.DISABLED)

def evaluate_model(model, model_name):
    """
    Function to train the model, make predictions, and evaluate performance:
    - Calculate accuracy, precision, recall, and F1 score
    - Display confusion matrix heatmap
    - Update the text area with the evaluation results
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix heatmap
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()
    
    # Update text area with results
    result_text = (f"{model_name} results:\n"
                   f"Accuracy: {accuracy:.2f}\n"
                   f"Precision: {precision:.2f}\n"
                   f"Recall: {recall:.2f}\n"
                   f"F1 Score: {f1:.2f}\n"
                   f"Confusion Matrix:\n{cm}")
    update_text_area(result_text)

def train_decision_tree():
    """
    Function to train and evaluate a Decision Tree classifier.
    """
    dt = DecisionTreeClassifier(random_state=42)
    evaluate_model(dt, "Decision Tree")
    
    # Plot decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(dt, filled=True, feature_names=X.columns, class_names=['Benign', 'Malignant'])
    plt.show()

def knn():
    """
    Function to train and evaluate a K-Nearest Neighbors classifier.
    """
    knn = KNeighborsClassifier()
    evaluate_model(knn, "KNN")

def naive_bayes():
    """
    Function to train and evaluate a Naive Bayes classifier.
    """
    gnb = GaussianNB()
    evaluate_model(gnb, "Naive Bayes")

def svm():
    """
    Function to train and evaluate a Support Vector Machine classifier.
    """
    svm = SVC()
    evaluate_model(svm, "SVM")

def get_data_head():
    """
    Function to return the first 5 rows of the dataset.
    """
    return data.head()

def get_data_describe():
    """
    Function to return descriptive statistics of the dataset.
    """
    return data.describe()

def get_data_missing_values():
    """
    Function to return the number of missing values in each column.
    """
    return data.isnull().sum()

def get_histogram():
    """
    Function to plot histograms for the 'diagnosis' feature in the cleaned dataset.
    """
    df_cleaned['diagnosis'].hist(bins=30)
    plt.xlabel('Diagnosis')
    plt.ylabel('Frequency')
    plt.title('Histogram of Diagnosis')
    plt.show()

def correlation_matrix():
    """
    Function to plot a correlation matrix heatmap for the cleaned dataset.
    """
    plt.figure(figsize=(20, 15))
    correlation_matrix_cleaned = df_cleaned.corr()
    sns.heatmap(correlation_matrix_cleaned, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix Heatmap (Cleaned)')
    plt.show()

def Scatter():
    # make scatter plot of the data between 'radius_mean' and 'texture_mean'
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='radius_mean', y='texture_mean', data=df_cleaned, hue='diagnosis')
    plt.title('Scatter Plot of Radius Mean vs. Texture Mean')
    plt.show()
    pass

# Mapping of data methods to corresponding functions
data_methods = {
    'Data Head': get_data_head,
    'Data Describe': get_data_describe,
    'Data Missing Values': get_data_missing_values,
    'Data Cleaning': DataCleaning,
    'Histogram': get_histogram,
    'Correlation Matrix': correlation_matrix,
    'Scatter': Scatter,
    'Decision Tree': train_decision_tree,
    'KNN': knn,
    'Naive Bayes': naive_bayes,
    'SVM': svm
}

# Mapping of category operations to corresponding methods
category_operations = {
    'Preprocessing': ['Data Head', 'Data Describe', 'Data Missing Values', 'Data Cleaning'],
    'Graphs': ['Histogram', 'Correlation Matrix', 'Scatter'],
    'Classifiers': ['Decision Tree', 'KNN', 'Naive Bayes', 'SVM']
}

def update_operations(*args):
    """
    Function to update the operations combo box based on the selected category.
    """
    category = category_combo.get()
    operations_combo['values'] = category_operations.get(category, [])
    operations_combo.current(0)

def display_data():
    """
    Function to display the data or perform an operation based on the selected method.
    """
    method = operations_combo.get()
    if method in data_methods:
        result = data_methods[method]()
        if isinstance(result, pd.DataFrame) or isinstance(result, pd.Series):
            update_text_area(result.to_string())
        else:
            result()

# Initialize the main GUI window
root = tk.Tk()
root.title("Data Display GUI")
root.geometry('800x600')

# Category combo box
category_combo = ttk.Combobox(root, values=list(category_operations.keys()), state="readonly", width=50)
category_combo.current(0)
category_combo.pack(pady=20)
category_combo.bind("<<ComboboxSelected>>", update_operations)

# Operations combo box
operations_combo = ttk.Combobox(root, state="readonly", width=50)
operations_combo.pack(pady=10)
update_operations()

# Display button
display_button = ttk.Button(root, text="Display Data", command=display_data)
display_button.pack(pady=10)

# Text area with scrollbars
text_area = tk.Text(root, height=20, width=80, wrap=tk.NONE)
scroll_x = tk.Scrollbar(root, orient="horizontal", command=text_area.xview)
scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
text_area.config(xscrollcommand=scroll_x.set)
scroll_y = tk.Scrollbar(root, orient="vertical", command=text_area.yview)
scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
text_area.config(yscrollcommand=scroll_y.set)
text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
text_area.config(state=tk.DISABLED)

# Start the GUI main loop
root.mainloop()
