import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv('credit_customers (DS).csv')

# Drop duplicates
df_cleaned = df.drop_duplicates()

# Encode categorical variables
df_encoded = df_cleaned.copy()
categorical_cols = df_encoded.select_dtypes(include='object').columns.tolist()
categorical_cols.remove('class')  # target variable

# One-hot encoding
df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)

# Label encode the target variable
label_encoder = LabelEncoder()
df_encoded['class'] = label_encoder.fit_transform(df_encoded['class'])  # good=1, bad=0

# Feature and target split
X = df_encoded.drop('class', axis=1)
y = df_encoded['class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Results dictionary
results = {}

# Model evaluation function
def evaluate_model(name, model):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n{name} Results:")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
evaluate_model("Logistic Regression", lr)

# KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
evaluate_model("KNN Classifier", knn)

# SVM - Linear Kernel
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
evaluate_model("SVM Linear", svm_linear)

# SVM - RBF Kernel
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
evaluate_model("SVM RBF", svm_rbf)

# Best model
best_model = max(results, key=results.get)
print(f"\nBest Performing Model: {best_model} with Accuracy = {results[best_model]:.4f}")
