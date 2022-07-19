from email.headerregistry import Address
from operator import contains
import os
import csv
import numpy as np
import math
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pathlib
from pathlib import Path
import sys
import shutil
from random import random
from random import seed
seed(1)
from PIL import Image

# 각 감정 별로 몇대 몇으로 계속 나눠서 한다.
# 학습용 데이터 비율은 각각 0.65:0.15:0.2으로 한다. 데이터 값이 충분하니까.
# 학습용 부분들을 분리시키고



class EmoClassify():
    def __init__(self):
        dataAddress = "TrimmedSet"
        labelflie = open("TrimmedSet/labels.csv",'r')
        labelReader = csv.reader(labelflie)
        next(labelReader)
        #self.labelList = list(labelReader)
        #self.tp_labellist=[list(x) for x in zip(*self.labelList)] 
        #print(np.shape(self.labelList))
        #print(np.shape(self.tp_labellist))

        #self.CountGlobal = 1
        # 이거 기반으로 학습하는 중
        # https://colab.research.google.com/github/tensorflow/docs-l10n/blob/master/site/ko/tutorials/images/cnn.ipynb?hl=ko#scrollTo=MdDzI75PUXrG
        #print(np.shape(labelHeader))
        #self.labels = os.listdir(dataAddress)
        #self.labels.remove('labels.csv')
        #self.train_label_csv =  csv.writer(open("TrainSet/labels.csv",'w',newline=''))
        #self.test_label_csv =  csv.writer(open("TestSet/labels.csv",'w',newline=''))

        #for label in self.labels:
            #self.MoveForTrainTest(label)    
        #트레이닝 셋이 완성되게 되면 이이후에그값을 기반으로 이름 중간에 적어둔 데이터로 계산을 하게 된다.
        #self.SetLabelsAttheEndofProcess()
        self.SetLabelsto()

        #print("totalLength"+ str(self.CountGlobal))


        #이제 해당하는 데이터 값이 모였다. 그러면 기존의 sav파일은 어떻게 돌아갈 것인가? 
        # 

    def Findlabel(self,address):
        column_id = self.tp_labellist[0].index(address)
        #print (address + self.labelList[column_id][1])
        #print(": "+ str(self.tp_labellist[0].index(address)))
        return self.labelList[column_id][1]
    
    def SetLabelsAttheEndofProcess(self):
        trainAddress = "TrainSet/"
        trainData=os.listdir(trainAddress)
        trainData.remove('labels.csv')
        
        for file in trainData:
            tmp_strings = file.split('.')
            name = tmp_strings[0]
            tmp_strings = name.split('_')
            label = tmp_strings[len(tmp_strings)-1]
            self.train_label_csv.writerow([label])

        testAddress = "TestSet/"
        testData=os.listdir(testAddress)
        testData.remove('labels.csv')
        
        for file in testData:
            tmp_strings = file.split('.')
            name = tmp_strings[0]
            tmp_strings = name.split('_')
            label = tmp_strings[len(tmp_strings)-1]
            self.test_label_csv.writerow([label])

        testAddress = "TestSet/"
        testData = os.listdir(testAddress)
        testData.remove('labels.csv')
        labelflie = open("TestSet/labels.csv",'r')
        labelReader = csv.reader(labelflie)
        labelReaderList = list(labelReader)

        print(str(testData[0]) +"+"+ str(labelReaderList[0]))

    def SetLabelsto(self):
        trainAddress = "TrainSet/"
        trainData=os.listdir(trainAddress)
        trainData.remove('labels.csv')
        
        for file in trainData:
            path = os.path.join(trainAddress, file)
            if os.path.isdir(path):
                continue
            tmp_strings = file.split('.')
            name = tmp_strings[0]
            tmp_strings = name.split('_')
            label = tmp_strings[len(tmp_strings)-1]
            os.rename("TrainSet/"+file,"TrainSet/"+label +"/"+file)
            #self.train_label_csv.writerow([label])

        testAddress = "TestSet/"
        testData=os.listdir(testAddress)
        
        for file in testData:
            path = os.path.join(testAddress, file)
            if os.path.isdir(path):
                continue

            tmp_strings = file.split('.')
            name = tmp_strings[0]
            tmp_strings = name.split('_')
            label = tmp_strings[len(tmp_strings)-1]
            os.rename("TestSet/"+file,"TestSet/"+label +"/"+file)
            #self.test_label_csv.writerow([label])




            

        

    def MoveForTrainTest(self,label):
        # 여기서 할것이 학습 데이터 셋과 비학습 데이터 셋을 분리한다.
        # 분리하는 방법은 각 폴더 를 순회하면서 폴더 안의 파일의 수들을기반으로 각각 4:1로 나눈다.
        # 파일의 수를 체크하면서 앞에서부터 4/5 는 학습, 1/5는 검증으로 돌리며
        # 한 파일 한 파일 이동시키는 과정에서 수를 카운팅한다.  
        # 수를 카운트 할때 (여기서 머릿속에 에러가 났네, 도커 관련한 걱정때문에 잠시 아무튼)
        # 수를 카운팅하는것은 글로벌 수와 라벨별 값으로 분류 한다
        # 글로벌 수를 기반으로 이미지의 라벨을 새로운 trainning.csv에기입을 해두며
        # 라벨 별값 으로는 사이즈 측정하는 수준으로 
        emotionAddress = "TrimmedSet/"+label
        imgList = os.listdir(emotionAddress)
        trainlimit = math.floor(len(imgList) * 0.75 )
        print (str(self.CountGlobal)+","+ str(trainlimit) + "," + "," + str(len(imgList)))
        CountByLabel = 0

        for file in imgList:
            #shutil.move()
            #지금 문제가. 데이터호출 관련해서 내림 차순이 아니라 그냥 숫자가 0부터 시작하는 방식으로 호출중임.
            if CountByLabel < trainlimit:
                shutil.copy(emotionAddress +"/"+file, "TrainSet/")# to train
                tmp_label =self.Findlabel(label + "/"+file)
                #self.train_label_csv.writerow([tmp_label])
                strings = file.split('.')
                #print(tmp_label)
                os.rename("TrainSet/"+file,"TrainSet/"+strings[0]+"_" +tmp_label+"."+strings[1])
                #print(self.labelList.index[label + "/"+file])
                j=0
            else :
                j=0
                shutil.copy(emotionAddress +"/"+file, "TestSet/")# to test
                tmp_label =self.Findlabel(label + "/"+file)
                #self.test_label_csv.writerow([tmp_label])
                strings = file.split('.')
                #print(strings[1])
                os.rename("TestSet/"+file,"TestSet/"+strings[0]+"_" +tmp_label+"."+strings[1])

            self.CountGlobal = self.CountGlobal +1
            CountByLabel=CountByLabel +1
        
        #
        #print("Global check :"+ str(self.CountGlobal)+ " , this loop"+ str(CountByLabel))
            


def GetTrainImage():
    j=1

    return 1 
def GetTrainLabel():
    j=1

    return 1 
def pythonremovefast (filepath):
    if os.path.exists(filepath):
        for file in os.scandir(filepath):
            os.remove(file.path)
        print("kill all")
    else:
        print("nopath")
def pythonremoveimagefast (filepath):
    if os.path.exists(filepath):
        for file in os.scandir(filepath):
            if file.path.__contains__("png"):
                print("hi"+file.path)
                os.remove(file.path)
            elif file.path.__contains__("jpg"):
                print("hi"+file.path)
                os.remove(file.path)

        print("kill all")
    else:
        print("nopath")


def clearAllTestAndTrain():
   
    testAddress = "TestSet/"
    labels = os.listdir(testAddress)
    #for label in labels:
        #pythonremovefast(testAddress+label)
    pythonremovefast(testAddress)
    trainAddress = "TrainSet/"
    labelList = list(labels)
    #for label in labelList:
        #pythonremovefast(trainAddress+label)
    pythonremovefast(trainAddress)

def checkDataAndLabel():
    testAddress = "TrainSet/"
    testData = os.listdir(testAddress)
    testData.remove('labels.csv')
    labelflie = open("TrainSet/labels.csv",'r')
    labelReader = csv.reader(labelflie)
    labelReaderList = list(labelReader)
    for i in range(10):
        print(str(testData[i]) +"+"+ labelReaderList[i][0])


def configure_for_performance(ds,batch_size):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds    

if __name__ == "__main__":
    

    #clearAllTestAndTrain()
    #testAddress = "TestSet/"
    #pythonremoveimagefast(os.getcwd())
    #checkDataAndLabel()
    #self=EmoClassify()
    batch_size = 32
    img_height = 224
    img_width = 224
    train_dir = "TrainSet/"
    test_dir = "TestSet/"
    datas = os.listdir(train_dir)
    image_count = len(datas)
    #print(image_count)

    model = VGG16(weights='imagenet', include_top=True)
    #new_model.summary()
    #model.summary()
    #train_image_generator = ImageDataGenerator(rescale=1./255)
    #test_image_generator = ImageDataGenerator(rescale=1./255)
    #train_ds = train_image_generator.flow_from_directory(directory = train_dir, shuffle=True,target_size=(img_height, img_width),class_mode='binary')
    #vaild_ds = test_image_generator.flow_from_directory(directory = test_dir, shuffle=True,target_size=(img_height, img_width),class_mode='binary')
    
    train_ds=tf.keras.preprocessing.image_dataset_from_directory(train_dir,  seed=1337,image_size=(img_height, img_width),batch_size=batch_size)
    vaild_ds=tf.keras.preprocessing.image_dataset_from_directory(test_dir,  seed=1337,image_size=(img_height, img_width),batch_size=batch_size)

    #class_names = train_ds.class_names
    #print(class_names)
    #for image_batch, labels_batch in train_ds:
        #print(image_batch.shape)
        #print(labels_batch.shape)
        #break
    #train_ds = configure_for_performance(train_ds,batch_size)
    #val_ds = configure_for_performance(vaild_ds,batch_size)
    model.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    history = model.fit(train_ds, epochs=20,validation_data=vaild_ds)
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    print("acc : "+str(acc[len(acc)-1]))
    print("val_accuracy : "+str(val_acc[len(val_acc)-1]))
    print("loss : "+str(loss[len(loss)-1]))
    print("val_loss : "+str(val_loss[len(val_loss)-1]))
    model.save('OSW_VGGBasedModel_Generation')

    exit()






        











