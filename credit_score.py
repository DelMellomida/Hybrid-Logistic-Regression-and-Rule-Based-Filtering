import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load the dataset
df = pd.read_excel(r"a_Dataset_CreditScoring.xlsx")

# Dropping customer ID column from the dataset
df = df.drop('ID', axis=1)

# Step 1: Discard rows with more than 50% missing values
row_threshold = len(df.columns) / 2
df = df.dropna(thresh=row_threshold, axis=0)

# Step 2: Fill missing values
for col in df.columns:
    if df[col].dtype == 'object':  # Categorical variables
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:  # Continuous variables
        df[col].fillna(df[col].mean(), inplace=True)

# Step 3: Discard columns with more than 95% missing values
column_threshold = 0.95 * len(df)
df = df.loc[:, df.isnull().sum() < column_threshold]

# Verify cleaning process
print("\nFinal Missing Values:\n", df.isnull().sum())
print("\nDataset Shape After Cleaning:", df.shape)

# Step 4: Define target and features
y = df['TARGET']  # Target variable
X = df.drop(columns=['TARGET'])  # Features (excluding target)

# Identify categorical and continuous columns
categorical_columns = X.select_dtypes(include=['object']).columns
continuous_columns = X.select_dtypes(exclude=['object']).columns

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# Step 6: Standardization for continuous variables and encoding categorical variables
# Fit and transform OneHotEncoder on categorical features
ohe = OneHotEncoder()
ohe.fit(X_train[categorical_columns])
X_train_categorical = ohe.transform(X_train[categorical_columns]).toarray()
X_test_categorical = ohe.transform(X_test[categorical_columns]).toarray()

# Standardize continuous features
scaler = StandardScaler()
X_train_continuous = scaler.fit_transform(X_train[continuous_columns])
X_test_continuous = scaler.transform(X_test[continuous_columns])

# Combine continuous and categorical features
X_train_transformed = np.hstack([X_train_continuous, X_train_categorical])
X_test_transformed = np.hstack([X_test_continuous, X_test_categorical])

# Get column names
categorical_feature_names = ohe.get_feature_names_out(categorical_columns)
all_columns = np.concatenate([continuous_columns, categorical_feature_names])

# Save the preprocessor components
joblib.dump(scaler, 'scaler_CreditScoring.pkl')
joblib.dump(ohe, 'encoder_CreditScoring.pkl')

# Step 7: Train the Logistic Regression model
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train_transformed, y_train)

# Step 8: Predict on the test set
y_pred = classifier.predict(X_test_transformed)

# Step 9: Evaluate model performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 10: Save the trained model
joblib.dump(classifier, 'f1_Classifier_CreditScoring.pkl')

# Optional: Load the saved model for future predictions
# loaded_model = joblib.load('f1_Classifier_CreditScoring.pkl')
# y_new_pred = loaded_model.predict(X_new)  # X_new is new data to classify
