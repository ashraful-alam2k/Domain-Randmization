
import matlab.engine
import numpy as np

eng = matlab.engine.start_matlab()


# [X_des_train,X_in_train,Y_des_train,Y_in_train, X_des_test,X_in_test,Y_des_test,Y_in_test] = (eng.Sim_data_func(64,50,20,2,30,32768,16.8,1.14,0.21, 0.1,4.5,nargout=8))
Signal_x_after_fiber = (eng.Sim_data_func(64,50,20,2,30,32768,16.8,1.14,0.21, 0.1,4.5,nargout=1))


# print([X_des_train,X_in_train,Y_des_train,Y_in_train, X_des_test,X_in_test,Y_des_test,Y_in_test])
print(Signal_x_after_fiber)


