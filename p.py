import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns

data = pd.read_csv('breast-cancer.csv')

def DataCleaning():
    df = pd.read_csv('breast-cancer.csv')
    df.drop('id', axis=1, inplace=True)
    label_encoder = LabelEncoder()
    df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])
    features = df.columns.drop('diagnosis')
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    Q1 = df[features].quantile(0.25)
    Q3 = df[features].quantile(0.75)
    IQR = Q3 - Q1
    outlier_condition = ((df[features] < (Q1 - 1.5 * IQR)) | (df[features] > (Q3 + 1.5 * IQR))).any(axis=1)
    df_cleaned = df[~outlier_condition]
    return df_cleaned

df_cleaned = DataCleaning()
X = df_cleaned.drop('diagnosis', axis=1)
y = df_cleaned['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def update_text_area(text):
    text_area.config(state=tk.NORMAL)
    text_area.delete('1.0', tk.END)
    text_area.insert(tk.END, text)
    text_area.config(state=tk.DISABLED)

def train_decision_tree():
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    result_text = ("{Decision Tree} results:\n"
                   f"Accuracy: {accuracy:.2f}\n"
                   f"Precision: {precision:.2f}\n"
                   f"Recall: {recall:.2f}\n"
                   f"F1 Score: {f1:.2f}\n"
                   f"Confusion Matrix:\n{cm}")
    update_text_area(result_text)
    plt.figure(figsize=(20, 10))
    plot_tree(dt, filled=True, feature_names=X.columns, class_names=['Benign', 'Malignant'])
    plt.show()

def knn():
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    result_text = ("{KNN} results:\n"
                   f"Accuracy: {accuracy:.2f}\n"
                   f"Precision: {precision:.2f}\n"
                   f"Recall: {recall:.2f}\n"
                   f"F1 Score: {f1:.2f}\n"
                   f"Confusion Matrix:\n{cm}")
    update_text_area(result_text)

def naive_bayes():
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    result_text = ("{Naive Bayes} results:\n"
                   f"Accuracy: {accuracy:.2f}\n"
                   f"Precision: {precision:.2f}\n"
                   f"Recall: {recall:.2f}\n"
                   f"F1 Score: {f1:.2f}\n"
                   f"Confusion Matrix:\n{cm}")
    update_text_area(result_text)

def svm():
    svm = SVC()
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    result_text = ("{SVM} results:\n"
                   f"Accuracy: {accuracy:.2f}\n"
                   f"Precision: {precision:.2f}\n"
                   f"Recall: {recall:.2f}\n"
                   f"F1 Score: {f1:.2f}\n"
                   f"Confusion Matrix:\n{cm}")
    update_text_area(result_text)

def get_data_head():
    return data.head()

def get_data_describe():
    return data.describe()

def get_data_missing_values():
    return data.isnull().sum()

def get_histogram():
    df_cleaned.hist(bins=30, figsize=(20, 15))
    plt.show()

def correlation_matrix():
    plt.figure(figsize=(20, 15))
    correlation_matrix_cleaned = df_cleaned.corr()
    sns.heatmap(correlation_matrix_cleaned, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix Heatmap (Cleaned)')
    plt.show()

def perform_regression():
    # Placeholder function for regression
    pass

data_methods = {
    'Data Head': get_data_head,
    'Data Describe': get_data_describe,
    'Data Missing Values': get_data_missing_values,
    'Data Cleaning': DataCleaning,
    'Histogram': get_histogram,
    'Correlation Matrix': correlation_matrix,
    'Regression': perform_regression,
    'Decision Tree': train_decision_tree,
    'KNN': knn,
    'Naive Bayes': naive_bayes,
    'SVM': svm
}

category_operations = {
    'Preprocessing': ['Data Head', 'Data Describe', 'Data Missing Values', 'Data Cleaning'],
    'Graphs': ['Histogram', 'Correlation Matrix', 'Regression'],
    'Classifiers': ['Decision Tree', 'KNN', 'Naive Bayes', 'SVM']
}

def update_operations(*args):
    category = category_combo.get()
    operations_combo['values'] = category_operations.get(category, [])
    operations_combo.current(0)

def display_data():
    method = operations_combo.get()
    if method in data_methods:
        result = data_methods[method]()
        if isinstance(result, pd.DataFrame) or isinstance(result, pd.Series):
            update_text_area(result.to_string())
        else:
            result()

root = tk.Tk()
root.title("Data Display GUI")
root.geometry('800x600')

category_combo = ttk.Combobox(root, values=list(category_operations.keys()), state="readonly", width=50)
category_combo.current(0)
category_combo.pack(pady=20)
category_combo.bind("<<ComboboxSelected>>", update_operations)

operations_combo = ttk.Combobox(root, state="readonly", width=50)
operations_combo.pack(pady=10)
update_operations()

display_button = ttk.Button(root, text="Display Data", command=display_data)
display_button.pack(pady=10)

text_area = tk.Text(root, height=20, width=80, wrap=tk.NONE)
scroll_x = tk.Scrollbar(root, orient="horizontal", command=text_area.xview)
scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
text_area.config(xscrollcommand=scroll_x.set)
scroll_y = tk.Scrollbar(root, orient="vertical", command=text_area.yview)
scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
text_area.config(yscrollcommand=scroll_y.set)
text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
text_area.config(state=tk.DISABLED)

root.mainloop()
