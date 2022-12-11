# E-DBNN

params={"Resolution":2,"Number_of_Feature_Connections":2,"alpha":0.2}
Bx1=Bayesian_Model(params["Resolution"],params["Number_of_Feature_Connections"],params["alpha"])
Bx1.fit(X_train,Y_train,10)
print("Train Cost Factor",Bx1.Train_Cost_Factor)
Bx1.predict_data_set(X_test,Y_test)
print("Test Cost Factor",Bx1.Test_Cost_Factor)
