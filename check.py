import pandas as pd 

data = pd.read_csv("calendar_forecasting_corrected.csv")
for i in data.columns:
    print(type(i))
print(data.columns)
