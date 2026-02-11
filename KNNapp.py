import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------- Page Config --------------------
st.set_page_config(page_title="KNN Weather Classifier")
st.title("🌤 KNN Weather Classification App")

# -------------------- Dataset --------------------
X = np.array([
    [25, 80],
    [27, 60],
    [31, 65],
    [23, 85],
    [20, 75],
    [30, 70],
    [22, 90]
])

y = np.array([0, 1, 1, 0, 0, 1, 0])

label_map = {0: "Sunny", 1: "Rainy"}

# -------------------- Sidebar Input --------------------
st.sidebar.header("Input Features")
temp = st.sidebar.slider("Temperature (°C)", 15, 40, 26)
hum = st.sidebar.slider("Humidity (%)", 40, 100, 78)
k_value = st.sidebar.slider("Select K Value", 1, 5, 3)

# -------------------- Train/Test Split --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------- Model --------------------
knn = KNeighborsClassifier(n_neighbors=k_value)
knn.fit(X_train, y_train)

# -------------------- Prediction --------------------
new_data = np.array([[temp, hum]])
prediction = knn.predict(new_data)[0]

st.subheader("Prediction Result")
st.write(f"### Predicted Weather: **{label_map[prediction]}**")

# -------------------- Accuracy --------------------
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: **{accuracy:.2f}**")

# -------------------- Graph 1: Data Points --------------------
st.subheader("Training Data Visualization")

fig1, ax1 = plt.subplots()

ax1.scatter(X[y==0, 0], X[y==0, 1],
            color='orange', label='Sunny', s=100)

ax1.scatter(X[y==1, 0], X[y==1, 1],
            color='blue', label='Rainy', s=100)

ax1.scatter(temp, hum,
            color='red',
            marker='*',
            s=300,
            label='New Data')

ax1.set_xlabel("Temperature")
ax1.set_ylabel("Humidity")
ax1.legend()
ax1.grid(True)

st.pyplot(fig1)

# -------------------- Graph 2: Decision Boundary --------------------
st.subheader("Decision Boundary Visualization")

# Create mesh grid
h = 1
x_min, x_max = 15, 40
y_min, y_max = 40, 100

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig2, ax2 = plt.subplots()
ax2.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

ax2.scatter(X[y==0, 0], X[y==0, 1],
            color='orange', label='Sunny')

ax2.scatter(X[y==1, 0], X[y==1, 1],
            color='blue', label='Rainy')

ax2.scatter(temp, hum,
            color='black',
            marker='*',
            s=200,
            label='New Data')

ax2.set_xlabel("Temperature")
ax2.set_ylabel("Humidity")
ax2.legend()
ax2.grid(True)

st.pyplot(fig2)
