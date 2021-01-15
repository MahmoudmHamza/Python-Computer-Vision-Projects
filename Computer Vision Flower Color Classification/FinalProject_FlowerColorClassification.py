import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import listdir
from os.path import isfile , join
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#%%
df=pd.read_csv('flower_labels.csv')
data=df.drop([5, 15, 41, 44, 49, 61, 73, 80, 91, 94, 100, 104, 156, 158, 162, 176, 187, 192, 200, 209,2, 20, 26, 38, 60, 86, 93, 101, 105, 112, 121, 126, 131, 138, 144, 150, 163, 165, 178,6, 18, 25, 28, 29, 31, 55, 58, 66, 68, 76, 77, 88, 117, 129, 137, 160, 183, 186, 196, 198, 205, 208,13, 14, 22, 42, 46, 52, 62, 71, 90, 145, 152, 166, 179, 189, 191,37, 40, 53, 70, 72, 83, 113, 130, 140, 148, 154, 161, 167, 168, 170, 173, 185, 193])
img=data['file']
#%%
flower_targets = data['label'].values
images_files = [ join("C:\\Users\\MOHAMMED\\Desktop\\computerVision\\Task Final\\ماده رؤيه الحاسوب\\images", f) for f in listdir("images") if isfile(join("images" , f)) ]
#%%

def multi_View(images):
    images_count = len( images )
    fig = plt.figure(figsize=(10,20))
    for row in range( images_count  ):
        ax1 = fig.add_subplot( images_count , 1 , row + 1)    
        ax1.imshow( images[ row ] )
#%%        
def View(image):
    plt.figure(figsize=(10,20))
    plt.imshow( image )       
#%%                                                        
images = [ mpimg.imread( f ) for f in images_files ]
#%%
def Rgb2Gray(image):
    return np.dot(image[...,:3], [0.299, 0.587, 0.114])
#%%
grayImages = [Rgb2Gray(img) for img in images]
#%%
def histogram(img):
    height = img.shape[0]
    width = img.shape[1]
    hist = np.zeros((256))
    for i in np.arange(height):
        for j in np.arange(width):
            a = img.item(i,j)
            hist[int(a)] += 1
          
    return hist
#%%
def extract_color_stats(img):
	R,B,G=cv2.split(img)
	features = [np.mean(R), np.mean(G), np.mean(B), np.std(R),
		np.std(G), np.std(B)]
	return features 
#%%
def mean(img):
    
    mean=np.mean(img)
    return mean
Mean =[mean(img) for img in images]

Mean=np.hstack([Mean])
#%%
edges=[cv2.Canny(img,100,100)  for img in images] 
#%%
Hist = [histogram(img) for img in grayImages]
#%%
Feat = [extract_color_stats(img) for img in images] 
#%%
Features=np.hstack([Feat])
#%%
x_train, x_test, y_train, y_test = train_test_split(Features, flower_targets,test_size=0.20)

#%% '''Classification'''
##training a KNN classifier
knn = KNeighborsClassifier(n_neighbors=4).fit(x_train,y_train)
#accuracy
accuracyKNN =knn.score(x_test,y_test)
knnPredictions =knn.predict(x_test)
##creating confusion matrix
from sklearn.metrics  import confusion_matrix
cmKNN=confusion_matrix(y_test,knnPredictions)
print(accuracyKNN)

#%%
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

svmModelLinear = SVC(kernel = 'linear', C = 1).fit(x_train,y_train)
svmPredictions = svmModelLinear.predict(x_test)
 
# model accuracy for X_test  
accuracySVM = svmModelLinear.score(x_test,y_test)
print(accuracySVM)

 
# creating a confusion matrix
cmSVM = confusion_matrix(y_test, svmPredictions)


#%%
test=cv2.imread( "0116.jpg")
Hist_test = histogram(Rgb2Gray(test))  
Feat_test = extract_color_stats(test)
F_test=np.hstack([Feat_test])
prediction = svmModelLinear.predict(F_test.reshape(1,-1))[0]
# show predicted label on image
if prediction== 0:
    Prediction = ' 0 :-Phlox'
elif prediction== 4:
    Prediction = '4 :-calendula'
elif prediction== 5:
    Prediction = '5 :-iris    '
elif prediction== 3:
    Prediction = '3 :-Rose'
elif prediction== 8:
    Prediction = '8 :-Viola'    
    
    
while(1):
    cv2.putText(test,'Label= '+str(Prediction), (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255))
 # display the output image
    cv2.imshow('pic',test) 
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()
