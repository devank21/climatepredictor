import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
dataset=pd.read_csv('seattle-weather.csv')
x = dataset.iloc[:,1 :-1].values
y = dataset.iloc[:, -1].values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
def out(a, b, c,d):
  new_data = np.array([[a,b,c,d]])  # Create a NumPy array for new data
  scaled_data = sc.transform(new_data)  # Scale the new data using the scaler
  b= classifier.predict(scaled_data)  # Predict
  if b==0:
    return "drizzle"
  elif b==1:
    return "fog"
  elif b==2:
    return "rain"
  elif b==3:
    return "snow"
  elif b==4:
    return "sun"
  else:
    return "unpredictable weather.Please take care"


# [theme]
# backgroundColor = "#F0F0F0"


st.title(":red[Climate predictor]")
st.title("")
st.title("")
in1 = st.number_input("enter min_temperature")
st.text(" ")
in2 = st.number_input("Enter max_temperature")
st.text(" ")
in3 = st.number_input("enter precipitation")
st.text(" ")
in4 = st.number_input("Enter wind speed")
st.text(" ")

if st.button("Predict"):
    st.subheader("there is a possibilty of ")
    st.text(out(in1,in2,in3,in4))
    
    
    





