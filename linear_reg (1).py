
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
def cost(x,y,theta):
        
        r=(np.dot(x,theta))-y
        loss=0.5*(np.sum(r**2))
        return loss,r

def rmse(ypred,y):
        r=ypred-y
        r=np.sqrt(np.sum(r**2))
        return r/len(y)

def r2(ypred,y):
        sse= np.sum((ypred-y)**2)
        sst=np.sum((y-y.mean())**2)
        return 1-(sse/sst)

def gradient_descent(x,y,theta,L,epochs):
        cost_up=[]
        for i in range(epochs):    
                loss,error=cost(x,y,theta)
                cost_up.append(loss)
                dm = np.dot(x.T,error) 
                theta= theta - L*dm
        
        return theta,cost_up

def stochastic_gradesc(x,y,theta,L,epochs,batchsize=1):
        cost_up=[]
        for i in range(epochs):
                cost_sum=0
                for i in range(len(y)):
                        rand=np.random.randint(0,len(x))
                        loss,error=cost(x.values[rand],y.values[rand],theta)
                        dm = np.dot((x.values[rand]).T,error)
                        theta= theta - L*dm
                        cost_sum+=loss
                cost_up.append(cost_sum)
        
        return theta,cost_up

def mini_batch(x,y,theta,L,epochs,batchsize):
        cost_batch=[]
        m=len(y)/batchsize
        for i in range(epochs):
                cost_sum=0
                for i in range(0,len(y),batchsize):
                        rand=np.random.randint(0,len(x)-1,batchsize)
                        loss,error=cost(x.values[rand],y.values[rand],theta)
                        dm = np.dot((x.values[rand]).T,error)
                        theta= theta - L*(1/batchsize)*dm
                        cost_sum+=loss
                cost_batch.append(loss)
                
        return theta,cost_batch
                   
        
def main():
        data_set= pd.read_csv('C:/Users/lahar/OneDrive/Desktop/linear_reg/Admission_Predict_ver1.1.csv')# change the path to the path of the file in your system
        
        x = data_set[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
       'LOR ', 'CGPA', 'Research']]
        y = data_set['Chance of Admit ']
        x['intercept']=1
        norm=x.copy()
        n=len(x.columns)
        #normalising the features of the data set using min-max normalisation
        for feature_name in x.columns[:n-2] :
                max_value = x[feature_name].max()
                min_value = x[feature_name].min()
                norm[feature_name] = (x[feature_name] - min_value) / (max_value - min_value)
        
        x=norm
        m = np.zeros(x.shape[1])
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25) # splitting the data set into 300 images for training, 100 for testing
        
        #variables used for hyper parameter tetsing
        '''L = [0.001,0.004,0.008,0.02,0.04,0.06,0.08]
    
        epochs= [40,50,60,70,80,90,100,200,300,400,500]
        batchsize=[5,10,15,20,25,30,35,40,50,60,70]
        sse=[]
        
        Rmse=[]
        
        rsquare=[]
        for i in epochs:'''
                
        
        
              
        theta,cost_up= gradient_descent(x,y,m,0.001,epochs=90)
        c=cost(x_test,y_test,theta)
        sse=(c[0])           
        Rmse=(rmse(y_test,np.dot(x_test,theta)))         
        rsquare=(r2(y_test,np.dot(x_test,theta)))
        
        print("Gradient Descent")
        print("at 90 epochs and 0.001 alpha the cost is", sse)
        print("at 90 epochs and 0.001 alpha the RMSE is", Rmse)
        print("at 90 epochs and 0.001 alpha the R suqare is", rsquare)
        
        
        theta,cost_up= stochastic_gradesc(x,y,m,0.01,epochs=90)
        c=cost(x_test,y_test,theta)
        sse=(c[0])
        Rmse=(rmse(y_test,np.dot(x_test,theta)))        
        rsquare=(r2(y_test,np.dot(x_test,theta)))
        
        print("Stochastic Gradient Descent")
        print("at 90 epochs and 0.01 alpha the cost is", sse)
        print("at 90 epochs and 0.01 alpha the RMSE is", Rmse)
        print("at 90 epochs and 0.01 alpha the R square is", rsquare)
        
        theta,cost_up= mini_batch(x,y,m,0.07,epochs=90,batchsize=15)
        c=cost(x_test,y_test,theta)       
        sse=(c[0])         
        Rmse=(rmse(y_test,np.dot(x_test,theta)))         
        rsquare=(r2(y_test,np.dot(x_test,theta)))
        print("Mini-batch Gradient Descent")
        print("at 90 epochs,batch size of 15 and 0.07 alpha the cost is", sse)
        print("at 90 epochs, batch size of 15 and 0.07 alpha the RMSE is", Rmse)
        print("at 90 epochs, batch size of 15 and 0.07 alpha the R square is", rsquare)
                
                
                
                
                
               
              
       
        # graphs for report
        '''plt.plot(epochs, Rmse, label = "gradient descent") 
        #plt.plot(L, rsquare1, label = "sgd")
        #plt.plot(batchsize, rsquare, label = "mini batch")
        
        plt.xlabel('epoch')

        plt.ylabel('rsquare')

        plt.title('epoch vs rmse')

        plt.legend()
        plt.show()'''
        
      
        
        
if __name__ == "__main__":
    main()
