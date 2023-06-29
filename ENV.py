import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
import tensorflow as tf
from keras.datasets import fashion_mnist, mnist, cifar10
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import time
from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from keras import backend as K
class NAS():
    def __init__(self):
            self.done = False
            self.reward = 0
            self.acc = 0
            self.acc_old = 0.85
            self.acc_max = 0
            self.reward_max = 0
            self.structure_layers = 4
            self.kernal_size = [5 for _ in range(self.structure_layers)]    # 1,3,5,7,9
            self.filters_num = [32 for _ in range(self.structure_layers)]   # 6~64
            self.state = [self.kernal_size,self.filters_num]



    def step(self, action, ep, step, max_ep):
            
            if ep ==0:       
                self.state = [self.kernal_size, self.filters_num]
                print(self.state)
                self.reward, accuracy = self.run_NAS(ep, step, max_ep)
            else:
                # change kernal_size
                for k in range(self.structure_layers):
                    #### change kernal_size ###
                    if action == k:
                        self.kernal_size[k] += 2
                    elif action== (k + self.structure_layers):
                        self.kernal_size[k] -= 2

                    #### change filter_num ####
                    elif action== (k + 2*self.structure_layers):
                        self.filters_num[k] += 1
                    elif action== (k + 3*self.structure_layers):
                        self.filters_num[k] -= 1
                    #### do nothing ###
                    elif action==(4*self.structure_layers):
                        self.filters_num[k]=self.filters_num[k]
                        self.kernal_size[k]=self.kernal_size[k]
                self.state = [self.kernal_size, self.filters_num]
                # print(self.kernal_size)
                # print(self.filters_num)
                print(self.state)
                # Boundary check
                for ele in range(self.structure_layers):
                    if (self.kernal_size[ele]<1 or self.kernal_size[ele]>9 or self.filters_num[ele]<6 or self.filters_num[ele]>64):
                        print('Out of range')
                        self.done = True
                        break
                if self.done == True:
                    self.reward=-20
                    accuracy=0
                else:
                    self.reward, accuracy = self.run_NAS(ep, step, max_ep)
                     

            return self.reward, self.state, self.done, accuracy

            

    def run_NAS(self, ep, step, max_ep):
            baseline_list=[]
            def Build_Model_and_train(kernal_size,filter_num):
                batchsize = 32
                epochs_num = 10
                Dataset = cifar10
                customized_dataset = True
                if customized_dataset ==True:
                    data_dir = 'D:/sclab/LEDA/NAS_by_RL/CLS/train'
                    data_dir = Path(data_dir)
                    train_data_gen = ImageDataGenerator(rescale = 1./255,
                                                        shear_range = 0.2,
                                                        zoom_range = 0.2,
                                                        validation_split = 0.2,
                                                        horizontal_flip=True
                                                        )
                    train_dataset = train_data_gen.flow_from_directory(
                        directory=data_dir,
                        target_size=(64,64),
                        batch_size=32,
                        shuffle=True,
                        class_mode='categorical',
                        subset='training'
                    )

                    test_dataset = train_data_gen.flow_from_directory(
                        directory=data_dir,
                        target_size=(64,64),
                        batch_size=32,
                        shuffle=True,
                        class_mode='categorical',
                        subset='validation'
                    )
                # Load data 
                if Dataset == mnist:
                    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
                elif Dataset == fashion_mnist:
                    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
                elif Dataset == cifar10:
                    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

                # data pre-processing, data_format='channels_last'(避免error)
                if Dataset == mnist or Dataset == fashion_mnist :
                    X_train = X_train.reshape(-1, 28, 28, 1)
                    X_test = X_test.reshape(-1, 28, 28, 1)
                    Y_train = np_utils.to_categorical(Y_train, 10)
                    Y_test = np_utils.to_categorical(Y_test, 10)
                    inputshape = (28, 28, 1)
                    class_num=10
                elif Dataset == cifar10 :
                    X_train = X_train.astype('float32')/255
                    X_test = X_test.astype('float32')/255
                    Y_train = np_utils.to_categorical(Y_train, 10)
                    Y_test = np_utils.to_categorical(Y_test, 10)
                    inputshape = (32, 32, 3)
                    class_num=10
                if customized_dataset == True:
                    inputshape = (64, 64, 3)
                    class_num=18


                

                #build Model
                tf.compat.v1.reset_default_graph()
                K.clear_session()
                Model= Sequential()

                #layers
                for l in range(self.structure_layers):
                    Model.add(Convolution2D(filter_num[l], (kernal_size[l], kernal_size[l]), input_shape = inputshape, padding = "same", data_format='channels_last'))
                    Model.add(Activation('relu'))
                    if  l % 2 ==1:
                        Model.add(MaxPooling2D(pool_size = (2, 2), padding = "same"))
                    elif l == (self.structure_layers-1) and l%2 == 0:
                        Model.add(MaxPooling2D(pool_size = (2, 2), padding = "same"))
                    
                    
                # Fully Connected1 
                Model.add(Flatten())
                Model.add(Dense(1024))
                Model.add(Activation('relu'))
                

                # Fully Connected2 to (10)
                Model.add(Dense(class_num))
                Model.add(Activation('softmax'))


                # Model Summary
                print(Model.summary())


                # optimizer
                adam = Adam(learning_rate = 1e-4)

                # compile
                Model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

                print("-----------------Traning------------------")
                # Timer
                start = time.perf_counter()
                # Train
                if customized_dataset == True:
                    history = Model.fit(train_dataset, batch_size=batchsize, epochs=epochs_num,)
                else:
                    history = Model.fit(X_train, Y_train, batch_size=batchsize, epochs=epochs_num,)

                # # plt.plot(history.history['loss'])
                # # plt.title('Model Loss')
                # # plt.ylabel('loss')
                # # plt.xlabel('epoch')
                # # plt.legend(['train'], loc='upper right')
                # # plt.savefig('./Loss.png')
                # # plt.show()

                # # plt.plot(history.history['accuracy'])
                # # plt.title('Model Accuracy')
                # # plt.ylabel('accuracy')
                # # plt.xlabel('epoch')
                # # plt.legend(['train'], loc='upper left')
                # # plt.savefig('./Acc.png')
                # # plt.show()

                print('\n---------------Testing-----------------')
                if customized_dataset==True:
                    test_loss, test_accuracy = Model.evaluate(test_dataset)
                else:
                    test_loss, test_accuracy = Model.evaluate(X_test,Y_test)
                Time_required = time.perf_counter() - start
                print('Time required: ',Time_required)
                print('\ntest loss: ', test_loss)
                print('\ntest accuracy: ', test_accuracy)



                # print('\n---------------Saving Model-----------------')
                # Model.save('./CNN_classfier.h5')
                
                return Model, test_loss, test_accuracy , Time_required
            if ep ==0:
                f = open('Result/result.txt', 'a')
                f.write('ep = ' + str(ep)+'\n')
                baselineModel, baselinetest_loss, baselinetest_accuracy , baselineTime = Build_Model_and_train(self.kernal_size,
                                                                                                               self.filters_num)
                self.acc = baselinetest_accuracy
                f.write("BaselineTime: {}\n".format(baselineTime))
                f.write("BaselineAccuracy: {}\n".format(self.acc))
                f.close()
            elif ep > 0:
                f = open('Result/result.txt', 'a')
                f.write('ep = ' + str(ep) + ' step = ' + str(step) + '\n')
                f.write("kernal size for each layer: {} \n".format(self.state[0]))
                f.write("filters num for each layer: {} \n".format(self.state[1]))
                f.close()
                #讀取BASELINE
                f = open('Result/result.txt')
                for line in f.readlines():
                    if 'BaselineTime' in line:
                        baseline_list.append(line) 
                    elif 'BaselineAccuracy' in line:
                        baseline_list.append(line)
                f.close()
                for i in range(len(baseline_list)):
                    baseline_list[i] = baseline_list[i].split(':')[1]
                    baseline_list[i] = baseline_list[i].replace('\n','')
                    baseline_list[i] = baseline_list[i].replace(' ','')

                baseline = [float(i) for i in baseline_list]
                ###
                baselineAccuracy = baseline[1]
                baselineTime = baseline[0]
                print('baselineAccuracy=',str(baselineAccuracy))
                print('Baselinetime=',str(baselineTime))
                ###

                
                
                     
                Model, loss, accuracy, Time  = Build_Model_and_train(self.kernal_size, self.filters_num)
                
                #accuracy2,loss_history2 = Benchmarking.Benchmarking('mnist', [0,1], ['circuits'], [15], ['pca8'], circuit='QCNN', cost_fn= 'cross_entropy', binary=False)
                self.acc = accuracy ##計算reward使用
                #self.acc = accuracy1 ##計算reward使用
                self.reward = 100*(self.acc -0.80) - 5*(Time/baselineTime)
                f = open('Result/result.txt', 'a')
                f.write(str(loss))
                f.write("\n")
                #f.write(str(loss_history2))
                #f.write("\n")
                f.write("reward: {} \n".format(self.reward))
                f.write("accuracy: {}\n".format(self.acc))
                f.close()
                print("accuracy: {} ".format(self.acc))
                
                
                print("kernal size for each layer: {} \n".format(self.state[0]))
                print("filter num for each layer: {} \n".format(self.state[1]))
                print("reward: {} ".format(self.reward))

                if self.acc > self.acc_max:
                    self.acc_max = self.acc
                    f = open('bestuniformity.txt', 'w')
                    f.write('-------------------------------\nep = ' + str(ep) + ' step = ' + str(step) + '\n')
                    f.write("kernal size for each layer: {} \n".format(self.state[0]))
                    f.write("filter num for each layer: {} \n".format(self.state[1]))
                    f.write("reward: {} \n".format(self.reward))
                    f.write("accuracy: {} \n".format(self.acc))
                    # save model
                    Model.save('./CNN_classfier_bestAcc.h5')
                if self.reward > self.reward_max:
                    self.reward_max = self.reward
                    f = open('bestuniformity.txt', 'w')
                    f.write('-------------------------------\nep = ' + str(ep) + ' step = ' + str(step) + '\n')
                    f.write("kernal size for each layer: {} \n".format(self.state[0]))
                    f.write("filter num for each layer: {} \n".format(self.state[1]))
                    f.write("reward: {} \n".format(self.reward))
                    f.write("accuracy: {} \n".format(self.acc))
                    Model.save('./CNN_classfier_bestReward.h5')
                del Model
            self.acc_old = self.acc
                    
            return self.reward, self.acc_old


    

        

    def reset(self):
            self.done = False
            self.reward = 0
            self.kernal_size = [5 for _ in range(self.structure_layers)]
            self.filters_num = [32 for _ in range(self.structure_layers)]     
            self.state = [self.kernal_size,self.filters_num]
            self.acc_old = 0.65


            return self.state
# def SaveModel(layers_num):
# # Save Model
#     print('\n---------------Saving Model-----------------')
#     Model, loss, accuracy = Build_Model_and_train(layers_num)
#     Model.save('./CNN_classfier.h5')






 






# # Load Model
# Model = load_model('./CNN_classfier.h5')
# print('\n---------------Loading Model-----------------')
# print(Model.summary())




# one_img_predict(500)  
