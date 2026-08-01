import pandas as pd
def load_data():
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors = "coerce")
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print("\nShape: ",df.shape)
    
