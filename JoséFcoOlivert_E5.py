import json
import cv2
import os
import base64
import numpy as np
import random
from sklearn.feature_extraction import image
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
import sys



pathJson_train = f'{sys.argv[1]}' 
train = [fJson for fJson in os.listdir(pathJson_train) if fJson.endswith('.json')]

pathJson_test = f'{sys.argv[2]}' 
test = [fJson for fJson in os.listdir(pathJson_test) if fJson.endswith('.json')]




def rotate(image, angle, center=None, scale=1.0):
    # Extraemos las dimensiones de la imagen.
    (h, w) = image.shape[:2]

    # Por defecto, estableceremos el punto de rotación en el centro de la imagen.
    if center is None:
        center = (w // 2, h // 2)

    # Calculamos la matriz de rotación con base al centro, ángulo y escala.
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # Llevamos a cabo la rotación.
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h))

    return rotated

bufferData = []
Y=[]
pSize=(10,10)
aux=[]
angles=[3,7,13,-3,-7,-13]
for fJ in train:
    
    with open(pathJson_train+"/"+fJ) as infile:
        data = json.load(infile)
        
        
        for item in data["shapes"]:
             label = item["label"];Y.append(label)
        imdata = base64.b64decode(data["imageData"])
        npimg = np.frombuffer(imdata, dtype=np.uint8);
        im0 = cv2.imdecode(npimg, 1)  
        im0 = cv2.resize(im0,(100,100))   


        for i in range(len(angles)):
                Y.append(label)
                rotated=rotate(im0,angles[i])
                patches1 = image.extract_patches_2d(rotated, pSize, max_patches=225,random_state=0)
                patches1 = np.reshape(patches1, (len(patches1), -1))
                aux=np.append(aux,patches1)
            
        patches = image.extract_patches_2d(im0, pSize,max_patches=225,random_state=0)
        patches = np.reshape(patches, (len(patches), -1))
        aux=np.append(aux,patches)
        
        
        
        
        

aux=aux.reshape(-1,300)

aux1=[]


Y_test=[]

for fJ in test:
    
    with open(pathJson_test+"/"+fJ) as infile:
        data = json.load(infile)
        
        for item in data["shapes"]:
            label = item["label"]; Y_test.append(label)
        imdata = base64.b64decode(data["imageData"])
        npimg = np.frombuffer(imdata, dtype=np.uint8);
        im0 = cv2.imdecode(npimg, 1)
        
        im0 = cv2.resize(im0,(100,100)) 
            
        
        patches = image.extract_patches_2d(im0, pSize,max_patches=225,random_state=0)
        aux1=np.append(aux1,patches)
        
aux1=np.reshape(aux1,(-1,300)) 


kmeans = MiniBatchKMeans(n_clusters=40, verbose=False,random_state=0).fit(aux)
Y=np.array(Y)

X=np.reshape(kmeans.labels_,(len(Y),-1))
x=kmeans.predict(aux1)
X_test=np.reshape(x,(len(test),-1))



knn= KNeighborsClassifier(n_neighbors=2,weights="distance").fit(X,Y)

labels=knn.predict(X_test)



string=""""""
for i in range(len(labels)):
    index_str = str(i).zfill(4)
    label_str = str(labels[i])
    string+=(f"{index_str} {label_str}\n")

print(string)