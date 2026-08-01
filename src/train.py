from preprocess import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import joblib

df = load_data()
df.drop("customerID",axis = 1,inplace = True)
le = LabelEncoder()

for col in df.columns:
    if str(df[col].dtype) == "str":
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
    
X = df.drop("Churn",axis = 1)
y = df["Churn"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 42)

print("Train shape: ",X_train.shape)
print("Test shape: ",X_test.shape)

print(df.dtypes)
print(df.head())
model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy: ",accuracy)
  
  
tree = DecisionTreeClassifier(random_state =42)
tree.fit(X_train,y_train)
tree_pred = tree.predict(X_test)
tree_accuracy = accuracy_score(y_test,tree_pred)
print("Decision tree accuracy: ",tree_accuracy)

joblib.dump(model,"models/logistic_model.pkl")
print("Model Saved Successfully!")
