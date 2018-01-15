import pandas as pd
import numpy as np
from numpy import *
import mibian
#http://code.mibian.net/
import matplotlib.pyplot as pp

index=pd.read_csv('VAR_MC_Data.csv')
#print(index.head(100))

#SPX=index.iloc[:,1]
SPX=index.loc[:,'SPX']
DJX=index.loc[:,'DJX']
VIX=index.loc[:,'VIX']
VXD=index.loc[:,'VXD']

#Question 1-a
log_SPX=np.log(SPX)-np.log(SPX.shift(1))
log_DJX=np.log(DJX)-np.log(DJX.shift(1))
log_VIX=np.log(VIX)-np.log(VIX.shift(1))
log_VXD=np.log(VXD)-np.log(VXD.shift(1))
log_SPX=log_SPX[1:201]
log_DJX=log_DJX[1:201]
log_VIX=log_VIX[1:201]
log_VXD=log_VXD[1:201]

lamda=np.zeros(200)
for i in range(0,199):
    lamda[i]=0.94**i

var_SPX=(1-0.94)*np.sum(lamda*log_SPX*log_SPX)
var_DJX=(1-0.94)*np.sum(lamda*log_DJX*log_DJX)
var_VIX=(1-0.94)*np.sum(lamda*log_VIX*log_VIX)
var_VXD=(1-0.94)*np.sum(lamda*log_VXD*log_VXD)

#cov = pd.DataFrame(columns=['SPX','DJX','VIX','VXD'])
cov_SPX_DJX=(1-0.94)*np.sum(lamda*log_SPX*log_DJX)
cov_SPX_VIX=(1-0.94)*np.sum(lamda*log_SPX*log_VIX)
cov_SPX_VXD=(1-0.94)*np.sum(lamda*log_SPX*log_VXD)
cov_DJX_VIX=(1-0.94)*np.sum(lamda*log_DJX*log_VIX)
cov_DJX_VXD=(1-0.94)*np.sum(lamda*log_DJX*log_VXD)
cov_VIX_VXD=(1-0.94)*np.sum(lamda*log_VIX*log_VXD)

y=np.array([[var_SPX,cov_SPX_DJX,cov_SPX_VIX,cov_SPX_VXD],
            [cov_SPX_DJX,var_DJX,cov_DJX_VIX,cov_DJX_VXD],
            [cov_SPX_VIX,cov_DJX_VIX,var_VIX,cov_VIX_VXD],
            [cov_SPX_VXD,cov_DJX_VXD,cov_VIX_VXD,var_VXD]])
print('Covariance matrix')
print(y)
#Question 1-b
annual_var_SPX=np.sqrt(252)*np.sqrt(var_SPX)
annual_var_DJX=np.sqrt(252)*np.sqrt(var_DJX)
annual_var_VIX=np.sqrt(252)*np.sqrt(var_VIX)
annual_var_VXD=np.sqrt(252)*np.sqrt(var_VXD)

# Question 2
c_SPX=mibian.Me([SPX[0], 1865, 0.25, 2.22, 35], callPrice=(49.5+50.1)/2)
p_SPX=mibian.Me([SPX[0], 1865, 0.25, 2.22, 35], putPrice=(55.5+56.1)/2)
c_DJX=mibian.Me([DJX[0]/100, 160, 0.25, 2.48, 35], callPrice=(3.85+4.05)/2)
p_DJX=mibian.Me([DJX[0]/100, 160, 0.25, 2.48, 35], putPrice=(4.7+4.9)/2)
print('implied Volatility')
print('SPX call:',c_SPX.impliedVolatility)
print('SPX put:',p_SPX.impliedVolatility)
print('DJX call:',c_DJX.impliedVolatility)
print('DJX put:',p_DJX.impliedVolatility)

# Question 3 & 4
SPX_initial=mibian.Me([SPX[0], 1865, 0.25, 2.22, 35], volatility=(c_SPX.impliedVolatility+p_SPX.impliedVolatility)/2)
DJX_initial=mibian.Me([DJX[0]/100, 160, 0.25, 2.48, 35], volatility=(c_DJX.impliedVolatility+p_DJX.impliedVolatility)/2)
initial_value=-50*SPX_initial.callPrice-50*SPX_initial.putPrice+550*DJX_initial.callPrice+550*DJX_initial.putPrice

print(initial_value)
iteration=1000
A=np.linalg.cholesky(y)
A=mat(A)
PL=np.zeros(iteration+1)
for i in range(0,iteration):
    e = np.random.standard_normal((4, 1))
    x=A*e
    SPXmc=SPX[1]*np.exp(x[0])
    DJXmc=DJX[1]/100*np.exp(x[1])
    VIXmc=(c_SPX.impliedVolatility+p_SPX.impliedVolatility)/2*np.exp(x[2])
    VXDmc=(c_DJX.impliedVolatility+p_DJX.impliedVolatility)/2*np.exp(x[3])

    SPX_montecarlo=mibian.Me([SPXmc, 1865, 0.25, 2.22, 35], volatility=VIXmc)
    DJX_montecarlo=mibian.Me([DJXmc, 160, 0.25, 2.48, 35], volatility=VXDmc)
    mc_value=-50*SPX_montecarlo.callPrice-50*SPX_montecarlo.putPrice+550*DJX_montecarlo.callPrice+550*DJX_montecarlo.putPrice
    PL[i]=100*(mc_value-initial_value)

# print(PL)
VaR=np.percentile(PL,5)
ES=np.mean(PL[PL<= VaR])

print('Monte Carlo VaR:',VaR)
print('Expected Shortfall:',ES)

