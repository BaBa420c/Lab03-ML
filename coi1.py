import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
from sklearn.model_selection import train_test_split as tts
from sklearn.neighbors import KNeighborsClassifier as KNN   
df=pd.read_csv('student_data.csv')
class_F = df[df["sex"] == "F"]
class_M = df[df["sex"] == "M"]

features = ["age", "studytime", "absences", "G3"]

num_F = class_F[features].values
num_M = class_M[features].values

def mean(X):
    return np.sum(X, axis=0)/X.shape[0]
def variance(X):
    mu = mean(X)
    return np.sum((X - mu) ** 2, axis=0) / X.shape[0]
def std(X):
    return np.sqrt(variance(X))
#centroid
centroid_F = mean(num_F)
centroid_M = mean(num_M)
#spread
spread_F = std(num_F)
spread_M = std(num_M)
#euclidien distance 
distance = np.linalg.norm(centroid_F - centroid_M)
#output
print("Features:", features)
print("\nFemale Centroid:", centroid_F)
print("Male Centroid:", centroid_M)
print("\nFemale Spread:", spread_F)
print("Male Spread:", spread_M)
print("\nInterclass Distance:", distance)

#############
#print(num_F[:,2])
mean_absences_F = mean(num_F[:, 2])
variance_absences_F = variance(num_F[:, 2])
print("\nFemale Absences - Mean:", mean_absences_F)
print("Female Absences - Variance:", variance_absences_F)
plt.hist(num_F[:, 2], bins=10, alpha=0.5, label='Female', color='pink')
plt.title('Distribution of Female Students Grades')
plt.xlabel('no. of absences')
plt.ylabel('Number of Students')
#plt.show()

##########
def minkwoksi_dis(a,b,p):
    return np.sum(np.abs(a-b)**p)**(1/p)

for i in range(1,11):
    print("distance for p->",i,minkwoksi_dis(centroid_F,centroid_M,i))

##############
print(dist.minkowski(centroid_F,centroid_M,10))

############
# Use ALL students for classification
X = df[features].values  # All students' features
y = df["sex"].values     # All students' sex labels (F or M)

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3) 

########
neigh = KNN(n_neighbors=3) 
neigh.fit(X_train, y_train) 

# Make predictions and check accuracy
y_pred = neigh.predict(X_test)
accuracy = neigh.score(X_test, y_test)
print("\nKNN Classifier Accuracy:", accuracy) 