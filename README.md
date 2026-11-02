# fc

5th---------
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# Load
df = pd.read_csv("/content/drive/MyDrive/CS2225 Datasets for
lab/Program5_StudentsPerformance.csv")
X = df[['reading score']] # input
y = df['math score'] # output
# Split, train, predict
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
# Evaluate
print("Slope :", round(model.coef_[0], 3))
print("Intercept :", round(model.intercept_, 3))
print("MSE :", round(mean_squared_error(y_test, y_pred), 2))
print("R2 score :", round(r2_score(y_test, y_pred), 3))
# Predict a new value
new = pd.DataFrame({'reading score': [70]})
print("Predicted math score for reading score 70:", round(model.predict(new)[0], 2))


6th---------
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Load dataset
file_path = "/content/drive/MyDrive/CS2225 Datasets for lab/Program6_diabetes.csv"

try:
    df = pd.read_csv(file_path, encoding='latin1')
    print("Dataset Preview:\n", df.head())
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Please check the file path.")
    exit(1)
except pd.errors.EmptyDataError:
    print(f"Error: File '{file_path}' is empty or invalid.")
    exit(1)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit(1)


# Step 2: Check if required columns exist
required_columns = ['Glucose', 'Outcome']
if not all(col in df.columns for col in required_columns):
    print(f"Error: Required columns {required_columns} not found. Available columns: {list(df.columns)}")
    exit(1)


# Step 3: Prepare data
try:
    X = df[['Glucose']]  # Predictor
    y = df['Outcome']    # Target (0 = non-diabetic, 1 = diabetic)
except KeyError as e:
    print(f"Error accessing columns: {e}")
    exit(1)


# Step 4: Train/Test split
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
except Exception as e:
    print(f"Train/test split failed: {e}")
    exit(1)


# Step 5: Model training
try:
    model = LogisticRegression()
    model.fit(X_train, y_train)
except Exception as e:
    print(f"Model training failed: {e}")
    exit(1)


# Step 6: Prediction and Evaluation
try:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
except Exception as e:
    print(f"Prediction or evaluation failed: {e}")
    exit(1)


# Step 7: Predict for a new value
try:
    new_data = [[120]]  # Example Glucose level (mg/dL)
    predicted_result = model.predict(new_data)

    print(
        "Predicted Outcome for Glucose level 120 mg/dL:",
        "Diabetic" if predicted_result[0] == 1 else "Non-diabetic"
    )
except Exception as e:
    print(f"Prediction for new value failed: {e}")
    exit(1)



7th--------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Step 1: Load dataset
file_path = "/content/drive/MyDrive/CS2225 Datasets for lab/Program7_Iris.csv"

try:
    df = pd.read_csv(file_path, encoding='latin1')
    print("Dataset Preview:\n", df.head())
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Please check the file path.")
    exit(1)
except pd.errors.EmptyDataError:
    print(f"Error: File '{file_path}' is empty or invalid.")
    exit(1)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit(1)


# Step 2: Check if required columns exist
required_columns = [
    'sepal_length', 'sepal_width', 'petal_length',
    'petal_width', 'species'
]

if not all(col in df.columns for col in required_columns):
    print(f"Error: Required columns {required_columns} not found. Available columns: {list(df.columns)}")
    exit(1)


# Step 3: Prepare data
try:
    X = df.drop('species', axis=1)
    y = df['species']
except KeyError as e:
    print(f"Error accessing columns: {e}")
    exit(1)


# Step 4: Train/test split
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
except Exception as e:
    print(f"Train/test split failed: {e}")
    exit(1)


# Step 5: Model training
try:
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
except Exception as e:
    print(f"Model training failed: {e}")
    exit(1)


# Step 6: Predict and evaluate
try:
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)

    # Visualize confusion matrix
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=knn.classes_,
        yticklabels=knn.classes_
    )

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for K-NN')
    plt.show()

except Exception as e:
    print(f"Prediction or evaluation failed: {e}")
    exit(1)


# Step 7: Predict for new flower data
try:
    new_flower = [[5.0, 3.4, 1.5, 0.2]]
    result = knn.predict(new_flower)
    print("Prediction for new flower:", result[0])

except Exception as e:
    print(f"Prediction for new value failed: {e}")
    exit(1)



8th------
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# Step 1: Load dataset
file_path = "/content/drive/MyDrive/CS2225 Datasets for lab/Program8_Titanic.csv"

try:
    df = pd.read_csv(file_path, encoding='latin1')
    print("Dataset Preview:\n", df.head())
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Please check the file path.")
    exit(1)
except pd.errors.EmptyDataError:
    print(f"Error: File '{file_path}' is empty or invalid.")
    exit(1)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit(1)


# Step 2: Check if required columns exist
required_columns = [
    'PassengerId', 'Survived', 'Pclass', 'Sex', 'Age',
    'SibSp', 'Parch', 'Fare', 'Embarked'
]

if not all(col in df.columns for col in required_columns):
    print(f"Error: Required columns {required_columns} not found. Available columns: {list(df.columns)}")
    exit(1)


# Step 3: Feature selection and encoding
try:
    # Encode categorical variables
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # Select features (numerical and encoded categorical)
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

    X = df[features]
    y = df['Survived']

    # Handle missing values
    X = X.fillna(X.mean())

    if X.isnull().any().any():
        print("Error: Missing values remain after imputation.")
        exit(1)

except KeyError as e:
    print(f"Error accessing columns: {e}")
    exit(1)
except Exception as e:
    print(f"Error during preprocessing: {e}")
    exit(1)


# Step 4: Feature scaling
try:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
except Exception as e:
    print(f"Error during feature scaling: {e}")
    exit(1)


# Step 5: Apply PCA
try:
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print("Explained Variance Ratio:", pca.explained_variance_ratio_)
except Exception as e:
    print(f"Error during PCA transformation: {e}")
    exit(1)


# Step 6: Train/test split
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.3, random_state=42
    )
except Exception as e:
    print(f"Train/test split failed: {e}")
    exit(1)


# Step 7: Model training
try:
    model = LogisticRegression()
    model.fit(X_train, y_train)
except Exception as e:
    print(f"Model training failed: {e}")
    exit(1)


# Step 8: Predict and evaluate
try:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy after PCA:", accuracy)
except Exception as e:
    print(f"Prediction or evaluation failed: {e}")
    exit(1)


# Visualize PCA results
try:
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA of Titanic Dataset')
    plt.show()
except Exception as e:
    print(f"Visualization failed: {e}")
    exit(1)


# Step 9: Predict for new passenger data
try:
    # Example: [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]
    new_passenger = [[3, 0, 25.0, 1, 0, 7.25, 0]]

    new_passenger_scaled = scaler.transform(new_passenger)
    new_passenger_pca = pca.transform(new_passenger_scaled)
    result = model.predict(new_passenger_pca)

    print(
        "Prediction for new passenger:",
        "Survived" if result[0] == 1 else "Did not survive"
    )

except Exception as e:
    print(f"Prediction for new value failed: {e}")
    exit(1)


9th-------
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


# Step 1: Load dataset
file_path = "/content/drive/MyDrive/CS2225 Datasets for lab/Program9_Mall_Customers.csv"

try:
    df = pd.read_csv(file_path, encoding='latin1')
    print("Dataset Preview:\n", df.head())
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Please check the file path.")
    exit(1)
except pd.errors.EmptyDataError:
    print(f"Error: File '{file_path}' is empty or invalid.")
    exit(1)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit(1)


# Step 2: Check if required columns exist
required_columns = [
    'Annual Income (k$)', 'Spending Score (1-100)'
]

if not all(col in df.columns for col in required_columns):
    print(f"Error: Required columns {required_columns} not found. Available columns: {list(df.columns)}")
    exit(1)


# Step 3: Feature selection
try:
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

    # Handle missing values with mean imputation
    X = X.fillna(X.mean())

    if X.isnull().any().any():
        print("Error: Missing values remain after imputation.")
        exit(1)

except KeyError as e:
    print(f"Error accessing columns: {e}")
    exit(1)
except Exception as e:
    print(f"Error during feature selection: {e}")
    exit(1)


# Step 4: Feature scaling
try:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
except Exception as e:
    print(f"Error during feature scaling: {e}")
    exit(1)


# Step 5: Apply K-Means
try:
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
except Exception as e:
    print(f"Error during K-Means clustering: {e}")
    exit(1)


# Step 6: Evaluate
try:
    sil_score = silhouette_score(X_scaled, clusters)
    print("Silhouette Score:", sil_score)
except Exception as e:
    print(f"Error calculating silhouette score: {e}")
    exit(1)


# Step 7: Visualize clusters
try:
    scatter = plt.scatter(
        X_scaled[:, 0], X_scaled[:, 1],
        c=clusters, cmap='viridis', alpha=0.5
    )

    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        c='red', marker='x', s=200, label='Centroids'
    )

    plt.xlabel('Scaled Annual Income')
    plt.ylabel('Scaled Spending Score')
    plt.title('K-Means Clustering of Mall Customers')
    plt.colorbar(scatter, label='Cluster')
    plt.legend()
    plt.show()

except Exception as e:
    print(f"Visualization failed: {e}")
    exit(1)


# Step 8: Predict cluster for new customer
try:
    new_customer = [[50, 60]]

    new_customer_scaled = scaler.transform(new_customer)
    cluster = kmeans.predict(new_customer_scaled)

    print("Cluster for new customer:", cluster[0])

except Exception as e:
    print(f"Prediction for new customer failed: {e}")
    exit(1)
