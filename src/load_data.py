import pandas as pd
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(df.head())
print("\n Shape: ",df.shape)
print("\nColumns: ")
print(df.columns.tolist()) 

print("\n Dataset Info: ")
print(df.info())
print("\n Missing values: ")
print(df.isnull().sum())

print("\nDuplicate: ")
print(df.duplicated().sum())

print("\n Target Variable: ")
print(df["Churn"].value_counts())


df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")
print("\nMissing Values after Conversion: ")
print(df["TotalCharges"].isnull().sum()) 