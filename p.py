import tkinter as tk
from tkinter import ttk
import pandas as pd
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression       


# Load the dataset
data = pd.read_csv('student-por.csv')

# Preprocessing for demonstration purposes
data['age_squared'] = data['age'] ** 2  # Example feature engineering
features = ['age', 'absences', 'age_squared']  # Example features
data['pass'] = data['G3'] >= 10  # Example target based on a condition
X = data[features]
y = data['pass']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def train_and_evaluate(classifier, name):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    result_text = (f"{name} results:\n"
                   f"Accuracy: {accuracy:.2f}\n"
                   f"Precision: {precision:.2f}\n"
                   f"Recall: {recall:.2f}\n"
                   f"F1 Score: {f1:.2f}\n"
                   f"Confusion Matrix:\n{cm}")
    update_text_area(result_text)

def train_decision_tree():
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(X_train, y_train)
    y_pred = dt_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Display results in the text area
    result_text = f"Decision Tree Accuracy: {accuracy}\nConfusion Matrix:\n{cm}"
    update_text_area(result_text)
    
    # Optionally, plot the tree
    plt.figure(figsize=(20,10))
    tree.plot_tree(dt_classifier, filled=True, feature_names=features)
    plt.show()

def knn():
    train_and_evaluate(KNeighborsClassifier(), "KNN")

def naive_bayes():
    train_and_evaluate(GaussianNB(), "Naive Bayes")

def svm():
    train_and_evaluate(SVC(), "SVM")

# Decision Tree Classifier
def train_decision_tree():
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(X_train, y_train)
    y_pred = dt_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Display results in the text area
    result_text = f"Decision Tree Accuracy: {accuracy}\nConfusion Matrix:\n{cm}"
    update_text_area(result_text)
    
    # Optionally, plot the tree
    plt.figure(figsize=(20,10))
    tree.plot_tree(dt_classifier, filled=True, feature_names=features)
    plt.show()

# Define data functions
def get_data_head():
    return data.head().to_string()

def get_data_describe():
    return data.describe().to_string()

def get_data_missing_values():
    return data.isnull().sum().to_string()

def get_histogram():
    plt.hist(data['age'], bins=10, color='blue')
    plt.title('Distribution of Age')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()

def get_scatter_plot():
    plt.scatter(data['age'], data['G3'], alpha=0.5)
    plt.title('Scatter plot of Age vs G3')
    plt.xlabel('Age')
    plt.ylabel('Final Grade (G3)')
    plt.show()

def perform_clustering():
    kmeans = KMeans(n_clusters=3)
    data['cluster'] = kmeans.fit_predict(data[features])
    plt.scatter(data['age'], data['G3'], c=data['cluster'], cmap='viridis', alpha=0.5)
    plt.title('Clustering of Age vs G3')
    plt.xlabel('Age')
    plt.ylabel('Final Grade (G3)')
    plt.show()

def perform_regression():
    model = LinearRegression()
    X = data[['age']]
    y = data['G3']
    model.fit(X, y)
    predictions = model.predict(X)
    plt.scatter(data['age'], data['G3'], alpha=0.5)
    plt.plot(data['age'], predictions, color='red')
    plt.title('Linear Regression on Age vs G3')
    plt.xlabel('Age')
    plt.ylabel('Final Grade (G3)')
    plt.show()       

data_methods = {
    'Data Head': get_data_head,
    'Data Describe': get_data_describe,
    'Data Missing Values': get_data_missing_values,
    'Histogram': get_histogram,
    'Scatter Plot': get_scatter_plot,
    'Clustering': perform_clustering,
    'Regression': perform_regression,
    'Decision Tree': train_decision_tree,
    'KNN': knn,
    'Naive Bayes': naive_bayes,
    'SVM': svm
}

category_operations = {
    'Preprocessing': ['Data Head', 'Data Describe', 'Data Missing Values'],
    'Graphs': ['Histogram', 'Scatter Plot', 'Clustering', 'Regression'],
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
        update_text_area(result)

def on_search_entry_change(*args):
    search_query = search_var.get().title()
    matching_methods = [method for method in data_methods if search_query in method]
    autocomplete_menu.delete(0, tk.END)
    for method in matching_methods:
        autocomplete_menu.insert(tk.END, method)

def on_autocomplete_menu_select(event):
    selection = autocomplete_menu.get(autocomplete_menu.curselection())
    search_var.set(selection)
    result = data_methods[selection]()
    update_text_area(result)

def update_text_area(text):
    text_area.config(state=tk.NORMAL)
    text_area.delete('1.0', tk.END)
    text_area.insert(tk.END, text)
    text_area.config(state=tk.DISABLED)

root = tk.Tk()
root.title("Data Display GUI")
root.geometry('800x600')

# Category selector, previously lecture selector
category_combo = ttk.Combobox(root, values=list(category_operations.keys()), state="readonly", width=50)
category_combo.current(0)
category_combo.pack(pady=20)
category_combo.bind("<<ComboboxSelected>>", update_operations)

# Operations selector
operations_combo = ttk.Combobox(root, state="readonly", width=50)
operations_combo.pack(pady=10)
update_operations()  # Initialize with default category

# Display data button
display_button = ttk.Button(root, text="Display Data", command=display_data)
display_button.pack(pady=10)

# Search bar
search_var = tk.StringVar()
search_var.trace_add('write', on_search_entry_change)
search_entry = ttk.Entry(root, textvariable=search_var, width=53)
search_entry.pack(pady=10)

# Autocomplete listbox
autocomplete_menu = tk.Listbox(root, height=4)
autocomplete_menu.pack(pady=10)
autocomplete_menu.bind('<<ListboxSelect>>', on_autocomplete_menu_select)

# Text area for displaying results
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
