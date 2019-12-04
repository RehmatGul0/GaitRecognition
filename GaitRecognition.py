# Importing the libraries for data pre-processing
import numpy as np
import pandas as pd
##################################################################################################################################

#FUNCTIONS
# function to calculate min max

def min_and_max(iterable):
    min_value, max_value = float('inf'), float('-inf')
    for value in iterable:
        if value < min_value:
            min_value = value
        if value > max_value:
            max_value = value
    return min_value, max_value

# function to calculate linear time interpolation for successive accelaration wrt time
def interpolation(dataframe):
    minVal , maxVal = min_and_max(dataframe['time'].values)

    _dataframe = pd.DataFrame()
    _dataframe['itrp'] = np.interp(
        x=np.arange(start=minVal, stop=maxVal, step=100),
        xp=dataframe.iloc[:]['time'],
        fp=dataframe.iloc[:]['magnitude']
        )
    del minVal
    del maxVal
    return _dataframe

# function for framing datset creating features from seires
def frame(dataset_input,batch_size,batches,dataset,label):
    start = 0
    end = batch_size
    for index in range(2,batches+2):
        temp = dataset_input[start:end].tolist()
        temp.append(label)
        dataset = dataset.append(pd.Series(temp),ignore_index=True)
        start = end
        end = batch_size*index
    del index , start , end , temp
    return dataset

# function to calculate magnitude from from accelaration from x_axis,y_axis,z_axis
def magnitude(dataframe):
    dataframe['magnitude'] = ((dataframe['x'].values**2)+(dataframe['y'].values**2)+(dataframe['z'].values**2))**(0.5)
    return dataframe
##################################################################################################################################

# Importing the dataset
dataset_person_1_file_1 = pd.read_csv('.tsv',delimiter='\t',encoding='utf-8',names=['index','time','x','y','z','label'])
dataset_person_1_file_2 = pd.read_csv('.tsv',delimiter='\t',encoding='utf-8',names=['index','time','x','y','z','label'])
dataset_person_2_file_1 = pd.read_csv('.tsv',delimiter='\t',encoding='utf-8',names=['index','time','x','y','z','label'])
dataset_person_3_file_1 = pd.read_csv('.tsv',delimiter='\t',encoding='utf-8',names=['index','time','x','y','z','label'])
dataset_person_2_file_2 = pd.read_csv('.tsv',delimiter='\t',encoding='utf-8',names=['index','time','x','y','z','label'])
dataset_person_3_file_2 = pd.read_csv('.tsv',delimiter='\t',encoding='utf-8',names=['index','time','x','y','z','label'])

##################################################################################################################################
# calculating magnitude from from accelaration from x_axis,y_axis,z_axis
dataset_person_1_file_1 = magnitude(dataset_person_1_file_1)
dataset_person_1_file_2 = magnitude(dataset_person_1_file_2)
dataset_person_2_file_1 = magnitude(dataset_person_2_file_1)
dataset_person_3_file_1 = magnitude(dataset_person_3_file_1)
dataset_person_2_file_2 = magnitude(dataset_person_2_file_2)
dataset_person_3_file_2 = magnitude(dataset_person_3_file_2)
##################################################################################################################################

# calculating linear time interpolation for successive accelaration wrt time
dataset_person_1_file_1 = interpolation(dataset_person_1_file_1)
dataset_person_1_file_2 = interpolation(dataset_person_1_file_2)
dataset_person_3_file_1 = interpolation(dataset_person_3_file_1)
dataset_person_2_file_1 = interpolation(dataset_person_2_file_1)
dataset_person_3_file_2 = interpolation(dataset_person_3_file_2)
dataset_person_2_file_2 = interpolation(dataset_person_2_file_2)
##################################################################################################################################

#merging all datasets for p1 , p2 ,p3
dataset_person_1 = pd.concat([dataset_person_1_file_1,dataset_person_1_file_2])
dataset_person_2 = pd.concat([dataset_person_2_file_1,dataset_person_2_file_2])
dataset_person_3 = pd.concat([dataset_person_3_file_1,dataset_person_3_file_2])
del dataset_person_1_file_1
del dataset_person_1_file_2
del dataset_person_2_file_1
del dataset_person_2_file_2
del dataset_person_3_file_1
del dataset_person_3_file_2
##################################################################################################################################

#calculating size of minimum data available for preventing class imbalance and findix batch size for framing
min_size = dataset_p1.size if dataset_person_1.size < dataset_person_2.size else dataset_person_2.size
min_size = dataset_person_3.size if dataset_person_3.size < min_size else min_size
dataset_person_2 = dataset_person_2['itrp'].head(min_size)
dataset_person_1 = dataset_person_1['itrp'].head(min_size)
dataset_person_3 = dataset_person_3['itrp'].head(min_size)
del min_size
batch_size = 20
batches = int(dataset_person_1.size/batch_size)
##################################################################################################################################

# creating data frame having 30 steps as feature and label for rehmat's dataset
dataset = pd.DataFrame()
dataset=frame(dataset_person_1,batch_size,batches,dataset,0)
##################################################################################################################################

# creating data frame having 30 steps as feature and label for hassans's dataset
dataset= pd.concat([frame(dataset_person_2,batch_size,batches,dataset,1)])
##################################################################################################################################
 
# creating data frame having 30 steps as feature and label for hassans's dataset
dataset= pd.concat([frame(dataset_person_3,batch_size,batches,dataset,2)])
##################################################################################################################################

#Deleting unnecessary variables
del  dataset_person_2 , dataset_person_1 , dataset_person_3, batches
##################################################################################################################################

# creating numpy ndarray for machine learning algorithms input
x = dataset.iloc[:,0:batch_size].values
y =  dataset.iloc[:,-1].values
##################################################################################################################################

#importing libraries for classification and accuracy
import sklearn.metrics as metrics
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
##################################################################################################################################
    
#kfold validation with various algorithms
from sklearn.model_selection import KFold
# prepare cross validation
kfold = KFold(10, True, 1)
svm_description = []
nb_description = []
tree_description = []
random_forest_description = []

# enumerating 10 folds
for train, test in kfold.split(x):
    x_train , y_train = x[train] , y[train] 
    x_test , y_test = x[test] , y[test] 

    #svm 
    svm_model = svm.SVC(gamma='scale',probability=True,random_state=0)
    y_pred = svm_model.fit(x_train, y_train).predict(x_test)
    svm_description.append( {"Accuracy":"{0:.7f}".format(metrics.accuracy_score( y_test, y_pred)),
                             "Precission":"{0:.7f}".format(metrics.precision_score( y_test, y_pred,average='micro')),
                             "Recall":"{0:.7f}".format(metrics.recall_score( y_test, y_pred,average='micro'))
                             })
    
    #gaussian naive bayes
    gnb_model = GaussianNB()
    y_pred = gnb_model.fit(x_train, y_train).predict(x_test)
    nb_description.append( {"Accuracy":"{0:.7f}".format(metrics.accuracy_score( y_test, y_pred)),
                             "Precission":"{0:.7f}".format(metrics.precision_score( y_test, y_pred,average='micro')),
                             "Recall":"{0:.7f}".format(metrics.recall_score( y_test, y_pred,average='micro'))
                             })
    
    #decission tree
    tree_model = tree.DecisionTreeClassifier()
    y_pred = tree_model.fit(x_train, y_train).predict(x_test)
    tree_description.append( {"Accuracy":"{0:.7f}".format(metrics.accuracy_score( y_test, y_pred)),
                             "Precission":"{0:.7f}".format(metrics.precision_score( y_test, y_pred,average='micro')),
                             "Recall":"{0:.7f}".format(metrics.recall_score( y_test, y_pred,average='micro'))
                             })
    
    #random forest
    random_forest_model = RandomForestClassifier(n_estimators=30)
    y_pred = random_forest_model.fit(x_train, y_train).predict(x_test)
    random_forest_description.append( {"Accuracy":"{0:.7f}".format(metrics.accuracy_score( y_test, y_pred)),
                             "Precission":"{0:.7f}".format(metrics.precision_score( y_test, y_pred,average='micro')),
                             "Recall":"{0:.7f}".format(metrics.recall_score( y_test, y_pred,average='micro'))
                             })
##################################################################################################################################
    
from sklearn.cluster import KMeans
x=pd.DataFrame(x)
x['labels']=y
kmeans = KMeans(n_clusters=3, random_state=0).fit_predict(x.values)
x['kmeans_labels']=kmeans
cluster_0 = x[x['kmeans_labels'] == 0]
cluster0_rehmat = cluster_0[cluster_0['labels']==0]
cluster0_rehmat_count = cluster0_rehmat['labels'].count()

cluster0_hassan = cluster_0[cluster_0['labels']==1]
cluster0_hassan_count = cluster0_hassan['labels'].count()

cluster0_vjs = cluster_0[cluster_0['labels']==2]
cluster0_vjs_count = cluster0_vjs['labels'].count()



cluster_1 = x[x['kmeans_labels'] == 1]
cluster1_rehmat = cluster_1[cluster_1['labels']==0]
cluster1_rehmat_count = cluster1_rehmat['labels'].count()

cluster1_hassan = cluster_1[cluster_1['labels']==1]
cluster1_hassan_count = cluster1_hassan['labels'].count()

cluster1_vjs = cluster_1[cluster_1['labels']==2]
cluster1_vjs_count = cluster1_vjs['labels'].count()



cluster_2 = x[x['kmeans_labels'] == 2]
cluster2_rehmat = cluster_2[cluster_2['labels']==0]
cluster2_rehmat_count = cluster2_rehmat['labels'].count()

cluster2_hassan = cluster_2[cluster_2['labels']==1]
cluster2_hassan_count = cluster2_hassan['labels'].count()

cluster2_vjs = cluster_2[cluster_2['labels']==2]
cluster2_vjs_count = cluster2_vjs['labels'].count()

