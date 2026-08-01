from preprocess import load_data
import matplotlib.pyplot as plt

df=load_data()

df["Churn"].value_counts().plot(kind="bar")
plt.title("Customer churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.show()

pd = df.groupby(["gender","Churn"]).size().unstack()
pd.plot(kind = "bar")
plt.title("Gender vs Churn")
plt.xlabel("Gender")

plt.ylabel("Number of Customers")
plt.show()