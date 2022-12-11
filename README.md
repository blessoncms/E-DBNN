# E-DBNN

from Bayesian_Model import Bayesian_Model <br />
from Bayesian_Model import Bayesian_Model_PipeLine <br />
params={"Resolution":2,"Number_of_Feature_Connections":2,"alpha":0.2} <br />
Bx1=Bayesian_Model(params["Resolution"],params["Number_of_Feature_Connections"],params["alpha"]) <br />
Bx1.fit(X_train,Y_train,10) <br />
print("Train Cost Factor",Bx1.Train_Cost_Factor) <br />
Bx1.predict_data_set(X_test,Y_test) <br />
print("Test Cost Factor",Bx1.Test_Cost_Factor)<br />
