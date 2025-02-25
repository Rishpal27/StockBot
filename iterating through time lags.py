from Model_train import *
ma=0
l=[]
for i in range(1,60):
    result=100-arima(i)
    if result>ma:
        ma=result
        l.append(i)
print("The maximum acuuracy is :", ma," with the time lag: ",l[-1])
