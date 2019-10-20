#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
from pylab import plt
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')
import xlwt 
from xlwt import Workbook
import statistics
from scipy.stats import sem
from random import randint
from statistics import mean 
from scipy.stats import norm


# # Closed Form (Analytical) Valuations

# ### Binary Analytical Closed From Solution

# In[2]:


def Analytic_Binary_Call_Price(S, K, T, r, sigma):
    S = float(S)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    price = np.exp(-r * T) * norm.cdf(d2)
    
    return price


# In[3]:


Analytic_Binary_Call_Price(100,100,1,0.05,.20)


# In[4]:


def Analytic_Binary_Put_Price(S, K, T, r, sigma):
    S = float(S)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    price = np.exp(-r * T) * norm.cdf(-d2)
    
    return price


# In[5]:


Analytic_Binary_Put_Price(100,100,1,0.05,.20)


# ### Lookback Fixed Strike Analytical Closed From Solution

# In[6]:


#CALL
S=100
K=100
r=0.05
sigma=0.2
T=1

d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

AnalyticalLookbackFixedCall=S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2) + S*np.exp(-r*T)*((sigma**2)/(2*r))*(-(S/K)**((-2*r)/sigma**2)*norm.cdf(d1-(2*r*np.sqrt(T))/sigma)+np.exp(r*T)*norm.cdf(d1))
AnalyticalLookbackFixedCall


# In[7]:


#PUT
S=100
K=100
r=0.05
sigma=0.2
T=1

d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

AnalyticalLookbackFixedPut=K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1) + S*np.exp(-r*T)*((sigma**2)/(2*r))*((S/K)**((-2*r)/sigma**2)*norm.cdf(-d1+(2*r*np.sqrt(T))/sigma)-np.exp(r*T)*norm.cdf(-d1))
AnalyticalLookbackFixedPut


# ### Lookback Floating Strike Analytical Closed From Solution

# In[8]:


#CALL
S=100
Smin=100
r=0.05
sigma=0.2
T=1

d1 = (np.log(S / Smin) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
d2 = (np.log(S / Smin) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

AnalyticalLookbackFloatCall=S*norm.cdf(d1)-Smin*np.exp(-r*T)*norm.cdf(d2) + S*np.exp(-r*T)*((sigma**2)/(2*r))*((S/Smin)**((-2*r)/sigma**2)*norm.cdf(-d1+(2*r*np.sqrt(T))/sigma)-np.exp(r*T)*norm.cdf(-d1))
AnalyticalLookbackFloatCall


# In[9]:


#PUT
S=100
Smax=100
r=0.05
sigma=0.2
T=1

d1 = (np.log(S / Smax) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
d2 = (np.log(S / Smax) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

AnalyticalLookbackFloatPut=Smax*np.exp(-r*T)*norm.cdf(-d2)- S*norm.cdf(-d1)+ S*np.exp(-r*T)*((sigma**2)/(2*r))*(-(S/Smax)**((-2*r)/sigma**2)*norm.cdf(d1-(2*r*np.sqrt(T))/sigma)+np.exp(r*T)*norm.cdf(d1))
AnalyticalLookbackFloatPut


# # Monte Carlo Functions

# ### Regular Monte Carlo

# In[10]:


def MonteCarlo(S0,K,r,T,sigma,M,I,Payoff):
    dt=T/M  #length of time step
   
    #create two dimensional matrix of random std normal numbers (paths X timesteps)
    rn=np.random.standard_normal((M+1,I))
    
    #initialize S[0] to 100 stock price
    S=np.zeros_like(rn)
    S[0]=S0 
    
    #we want to iterate from 1 to M+1 so that the iteration reaches M, cus if we iterate till M, it will stop as soon as it reaches M without executing the M calculation.
    for t in range(1,M+1): 
        S[t]=S[t-1]*np.exp((r-sigma**2/2)*dt+(sigma*rn[t]*math.sqrt(dt)))
        
    if Payoff=='BinaryCall':
        PayoffBinaryCall=np.where(S[-1]>K,1,0)    
        #Now Discount the Average of all payoffs
        return math.exp(-r*T)*PayoffBinaryCall.mean()
    
    elif Payoff=='BinaryPut':
        PayoffBinaryPut=np.where(S[-1]<K,1,0)
        #Now Discount the Average of all payoffs
        return math.exp(-r*T)*PayoffBinaryPut.mean()
    
    elif Payoff=='LookFixedCall':
        PayoffLookFixedCall=np.zeros(I)
        for t in range(0,I):
            PayoffLookFixedCall[t]=np.maximum(max(S[:,t])-K,0)
        #Now Discount the Average of all payoffs
        return math.exp(-r*T)*PayoffLookFixedCall.mean()
    
    elif Payoff=='LookFixedPut':
        PayoffLookFixedPut=np.zeros(I)
        for t in range(0,I):
            PayoffLookFixedPut[t]=np.maximum(K-min(S[:,t]),0)
        #Now Discount the Average of all payoffs
        return math.exp(-r*T)*PayoffLookFixedPut.mean() 
    
    elif Payoff=='LookFloatCall':
        PayoffLookFloatCall=np.zeros(I)
        for t in range(0,I):
            PayoffLookFloatCall[t]=np.maximum(S[-1,t]-min(S[:,t]),0)
        #Now Discount the Average of all payoffs
        return math.exp(-r*T)*PayoffLookFloatCall.mean()
    
    elif Payoff=='LookFloatPut':
        PayoffLookFloatPut=np.zeros(I)
        for t in range(0,I):
            PayoffLookFloatPut[t]=np.maximum(max(S[:,t])-S[-1,t],0)
        #Now Discount the Average of all payoffs
        return math.exp(-r*T)*PayoffLookFloatPut.mean()


# ### Monte Carlo with Antithetic (AVT) and Moment Matching (MMT) Variance Reduction Techniques

# In[11]:


def MonteCarloAntitheticANDMomentMatch(S0,K,r,T,sigma,M,I,Payoff):
    dt=T/M  #length of time step
   
    #create two dimensional matrix of random std normal numbers (paths X timesteps)
    rn=np.random.standard_normal((M+1,I))
    
    #create the negative(inverse) version
    rn_minus=-rn

    #Standardize the positive Random Set(rn):
    rn=(rn-rn.mean())/rn.std()

    #Standardize the negative Random Set(rn):
    rn_minus=(rn_minus-rn_minus.mean())/rn_minus.std()
    
    #initialize S1[0] to 100 stock price
    S1=np.zeros_like(rn)
    S1[0]=S0 
    
    #initialize S2[0] to 100 stock price
    S2=np.zeros_like(rn)
    S2[0]=S0
    
    #Simulate Paths for S1
    for t in range(1,M+1): 
        S1[t]=S1[t-1]*np.exp((r-sigma**2/2)*dt+(sigma*rn[t]*math.sqrt(dt)))
        
    #Simulate Paths for S2
    for t in range(1,M+1): 
        S2[t]=S2[t-1]*np.exp((r-sigma**2/2)*dt+(sigma*rn_minus[t]*math.sqrt(dt)))
        
    if Payoff=='BinaryCall':
        PayoffBinaryCall1=np.where(S1[-1]>K,1,0)
        Call1=math.exp(-r*T)*PayoffBinaryCall1.mean()
        PayoffBinaryCall2=np.where(S2[-1]>K,1,0)
        Call2=math.exp(-r*T)*PayoffBinaryCall2.mean()
        #Now Average Both Option Values
        return (Call1+Call2)/2
    
    elif Payoff=='BinaryPut':
        PayoffBinaryPut1=np.where(S1[-1]<K,1,0)
        Put1=math.exp(-r*T)*PayoffBinaryPut1.mean()
        PayoffBinaryPut2=np.where(S2[-1]<K,1,0)
        Put2=math.exp(-r*T)*PayoffBinaryPut2.mean()
        #Now Average Both Option Values
        return (Put1+Put2)/2
    
    elif Payoff=='LookFixedCall':
        PayoffLookFixedCall1=np.zeros(I)
        for t in range(0,I):
            PayoffLookFixedCall1[t]=np.maximum(max(S1[:,t])-K,0)
        Call1=math.exp(-r*T)*PayoffLookFixedCall1.mean()
        PayoffLookFixedCall2=np.zeros(I)
        for t in range(0,I):
            PayoffLookFixedCall2[t]=np.maximum(max(S2[:,t])-K,0)
        Call2=math.exp(-r*T)*PayoffLookFixedCall2.mean()
        #Now Average Both Option Values
        return (Call1+Call2)/2
    
    elif Payoff=='LookFixedPut':
        PayoffLookFixedPut1=np.zeros(I)
        for t in range(0,I):
            PayoffLookFixedPut1[t]=np.maximum(K-min(S1[:,t]),0)
        Put1=math.exp(-r*T)*PayoffLookFixedPut1.mean()
        PayoffLookFixedPut2=np.zeros(I)
        for t in range(0,I):
            PayoffLookFixedPut2[t]=np.maximum(K-min(S2[:,t]),0)
        Put2=math.exp(-r*T)*PayoffLookFixedPut2.mean()
        #Now Average Both Option Values
        return (Put1+Put2)/2
    
    elif Payoff=='LookFloatCall':
        PayoffLookFloatCall1=np.zeros(I)
        for t in range(0,I):
            PayoffLookFloatCall1[t]=np.maximum(S1[-1,t]-min(S1[:,t]),0)
        Call1=math.exp(-r*T)*PayoffLookFloatCall1.mean()
        PayoffLookFloatCall2=np.zeros(I)
        for t in range(0,I):
            PayoffLookFloatCall2[t]=np.maximum(S2[-1,t]-min(S2[:,t]),0)
        Call2=math.exp(-r*T)*PayoffLookFloatCall2.mean()
        #Now Average Both Option Values
        return (Call1+Call2)/2        
        
    elif Payoff=='LookFloatPut':
        PayoffLookFloatPut1=np.zeros(I)
        for t in range(0,I):
            PayoffLookFloatPut1[t]=np.maximum(max(S1[:,t])-S1[-1,t],0)
        Put1=math.exp(-r*T)*PayoffLookFloatPut1.mean()
        PayoffLookFloatPut2=np.zeros(I)
        for t in range(0,I):
            PayoffLookFloatPut2[t]=np.maximum(max(S2[:,t])-S2[-1,t],0)
        Put2=math.exp(-r*T)*PayoffLookFloatPut2.mean()
        #Now Average Both Option Values
        return (Put1+Put2)/2


# # Part I: Binary Options

# ### Binary Option Valuation with 100,000 Simulations

# In[12]:


#Call
MonteCarlo(100,100,0.05,1.0,0.2,252,100000,'BinaryCall')


# In[13]:


#Put
MonteCarlo(100,100,0.05,1.0,0.2,252,100000,'BinaryPut')


# ### Binary Option Valuation with Different Number of Simulations

# In[14]:


# Workbook is created 
wb = Workbook()

# add_sheet is used to create sheet. 
sheet1 = wb.add_sheet('Sheet 1') 

z=-1 # Counter
for i in range(5000,1005000,5000):
    z=z+1
    
    sheet1.write(z, 1, MonteCarlo(100,100,0.05,1.0,0.2,252,i,'BinaryCall'))
wb.save('BinaryCall.xls')


# In[15]:


# Workbook is created 
wb = Workbook()

# add_sheet is used to create sheet. 
sheet1 = wb.add_sheet('Sheet 1') 

z=-1 # Counter
for i in range(5000,1005000,5000):
    z=z+1
    
    sheet1.write(z, 1, MonteCarlo(100,100,0.05,1.0,0.2,252,i,'BinaryPut'))
wb.save('BinaryPut.xls')


# ### Variance Reduction Techniques: Antithetic Variable and Moment Matching

# In[16]:


# Workbook is created 
wb = Workbook()

# add_sheet is used to create sheet. 
sheet1 = wb.add_sheet('Sheet 1') 

z=-1 # Counter
for i in range(5000,1005000,5000):
    z=z+1
    
    sheet1.write(z, 1, MonteCarloAntitheticANDMomentMatch(100,100,0.05,1.0,0.2,252,i,'BinaryCall'))
wb.save('BinaryCall.xls')


# In[17]:


# Workbook is created 
wb = Workbook()

# add_sheet is used to create sheet. 
sheet1 = wb.add_sheet('Sheet 1') 

z=-1 # Counter
for i in range(5000,1005000,5000):
    z=z+1
    
    sheet1.write(z, 1, MonteCarloAntitheticANDMomentMatch(100,100,0.05,1.0,0.2,252,i,'BinaryPut'))
wb.save('BinaryPut.xls')


# # Part II: Lookback Options

# ### Lookback Option Valuations with 100,000 Simulations

# In[18]:


#Fixed Strike Call
MonteCarlo(100,100,0.05,1.0,0.2,252,100000,'LookFixedCall')


# In[19]:


#Fixed Strike Put
MonteCarlo(100,100,0.05,1.0,0.2,252,100000,'LookFixedPut')


# In[20]:


#Floating Strike Call
MonteCarlo(100,100,0.05,1.0,0.2,252,100000,'LookFloatCall')


# In[21]:


#Floating Strike Put
MonteCarlo(100,100,0.05,1.0,0.2,252,100000,'LookFloatPut')


# ### Lookback Option Valuation with Different Number of Simulations

# ##### LOOKBACK FIXED STRIKE

# In[22]:


#CALL Regular
# Workbook is created 
wb = Workbook()

# add_sheet is used to create sheet. 
sheet1 = wb.add_sheet('Sheet 1') 

z=-1 # Counter
for i in range(5000,505000,5000):
    z=z+1
    
    sheet1.write(z, 1, MonteCarlo(100,100,0.05,1.0,0.2,252,i,'LookFixedCall'))
wb.save('FixedCall.xls')


# In[23]:


#Call AVT&MMT
# Workbook is created 
wb = Workbook()

# add_sheet is used to create sheet. 
sheet1 = wb.add_sheet('Sheet 1') 

z=-1 # Counter
for i in range(5000,505000,5000):
    z=z+1
    
    sheet1.write(z, 1, MonteCarloAntitheticANDMomentMatch(100,100,0.05,1.0,0.2,252,i,'LookFixedCall'))
wb.save('FixedCall.xls')


# In[24]:


#Put Regular
# Workbook is created 
wb = Workbook()

# add_sheet is used to create sheet. 
sheet1 = wb.add_sheet('Sheet 1') 

z=-1 # Counter
for i in range(5000,505000,5000):
    z=z+1
    
    sheet1.write(z, 1, MonteCarlo(100,100,0.05,1.0,0.2,252,i,'LookFixedPut'))
wb.save('FixedPut.xls')


# In[25]:


#Put AVT&MMT
# Workbook is created 
wb = Workbook()

# add_sheet is used to create sheet. 
sheet1 = wb.add_sheet('Sheet 1') 

z=-1 # Counter
for i in range(5000,505000,5000):
    z=z+1
    
    sheet1.write(z, 1, MonteCarloAntitheticANDMomentMatch(100,100,0.05,1.0,0.2,252,i,'LookFixedPut'))
wb.save('FixedPut.xls')


# ##### LOOKBACK FLOATING STRIKE

# In[26]:


#CALL Regular
# Workbook is created 
wb = Workbook()

# add_sheet is used to create sheet. 
sheet1 = wb.add_sheet('Sheet 1') 

z=-1 # Counter
for i in range(5000,505000,5000):
    z=z+1
    
    sheet1.write(z, 1, MonteCarlo(100,100,0.05,1.0,0.2,252,i,'LookFloatCall'))
wb.save('FloatCall.xls')


# In[27]:


#Call AVT&MMT
# Workbook is created 
wb = Workbook()

# add_sheet is used to create sheet. 
sheet1 = wb.add_sheet('Sheet 1') 

z=-1 # Counter
for i in range(5000,505000,5000):
    z=z+1
    
    sheet1.write(z, 1, MonteCarloAntitheticANDMomentMatch(100,100,0.05,1.0,0.2,252,i,'LookFloatCall'))
wb.save('FloatCall.xls')


# In[28]:


#Put Regular
# Workbook is created 
wb = Workbook()

# add_sheet is used to create sheet. 
sheet1 = wb.add_sheet('Sheet 1') 

z=-1 # Counter
for i in range(5000,505000,5000):
    z=z+1
    
    sheet1.write(z, 1, MonteCarlo(100,100,0.05,1.0,0.2,252,i,'LookFloatPut'))
wb.save('FloatPut.xls')


# In[29]:


#Put AVT&MMT
# Workbook is created 
wb = Workbook()

# add_sheet is used to create sheet. 
sheet1 = wb.add_sheet('Sheet 1') 

z=-1 # Counter
for i in range(5000,505000,5000):
    z=z+1
    
    sheet1.write(z, 1, MonteCarloAntitheticANDMomentMatch(100,100,0.05,1.0,0.2,252,i,'LookFloatPut'))
wb.save('FloatPut.xls')


# ### Lookback Option Valuation with Different Number of Time-Steps

# ##### LOOKBACK FIXED STRIKE

# In[30]:


#CALL Regular
# Workbook is created 
wb = Workbook()

# add_sheet is used to create sheet. 
sheet1 = wb.add_sheet('Sheet 1') 

z=-1 # Counter
for m in range(200,10200,200):
    z=z+1
    
    sheet1.write(z, 1, MonteCarlo(100,100,0.05,1.0,0.2,m,50000,'LookFixedCall'))
wb.save('FixedCall.xls')


# In[31]:


#Call AVT&MMT
# Workbook is created 
wb = Workbook()

# add_sheet is used to create sheet. 
sheet1 = wb.add_sheet('Sheet 1') 

z=-1 # Counter
for m in range(200,10200,200):
    z=z+1
    
    sheet1.write(z, 1, MonteCarloAntitheticANDMomentMatch(100,100,0.05,1.0,0.2,m,50000,'LookFixedCall'))
wb.save('FixedCall.xls')


# In[32]:


#Put Regular
# Workbook is created 
wb = Workbook()

# add_sheet is used to create sheet. 
sheet1 = wb.add_sheet('Sheet 1') 

z=-1 # Counter
for m in range(200,10200,200):
    z=z+1
    
    sheet1.write(z, 1, MonteCarlo(100,100,0.05,1.0,0.2,m,50000,'LookFixedPut'))
wb.save('FixedPut.xls')


# In[33]:


#Put AVT&MMT
# Workbook is created 
wb = Workbook()

# add_sheet is used to create sheet. 
sheet1 = wb.add_sheet('Sheet 1') 

z=-1 # Counter
for m in range(200,10200,200):
    z=z+1
    
    sheet1.write(z, 1, MonteCarloAntitheticANDMomentMatch(100,100,0.05,1.0,0.2,m,50000,'LookFixedPut'))
wb.save('FixedPut.xls')


# ##### LOOKBACK FLOATING STRIKE

# In[34]:


#CALL Regular
# Workbook is created 
wb = Workbook()

# add_sheet is used to create sheet. 
sheet1 = wb.add_sheet('Sheet 1') 

z=-1 # Counter
for m in range(200,10200,200):
    z=z+1
    
    sheet1.write(z, 1, MonteCarlo(100,100,0.05,1.0,0.2,m,50000,'LookFloatCall'))
wb.save('FloatCall.xls')


# In[35]:


#Call AVT&MMT
# Workbook is created 
wb = Workbook()

# add_sheet is used to create sheet. 
sheet1 = wb.add_sheet('Sheet 1') 

z=-1 # Counter
for m in range(200,10200,200):
    z=z+1
    
    sheet1.write(z, 1, MonteCarloAntitheticANDMomentMatch(100,100,0.05,1.0,0.2,m,50000,'LookFloatCall'))
wb.save('FloatCall.xls')


# In[36]:


#Put Regular
# Workbook is created 
wb = Workbook()

# add_sheet is used to create sheet. 
sheet1 = wb.add_sheet('Sheet 1') 

z=-1 # Counter
for m in range(200,10200,200):
    z=z+1
    
    sheet1.write(z, 1, MonteCarlo(100,100,0.05,1.0,0.2,m,50000,'LookFloatPut'))
wb.save('FloatPut.xls')


# In[37]:


#Put AVT&MMT
# Workbook is created 
wb = Workbook()

# add_sheet is used to create sheet. 
sheet1 = wb.add_sheet('Sheet 1') 

z=-1 # Counter
for m in range(200,10200,200):
    z=z+1
    
    sheet1.write(z, 1, MonteCarloAntitheticANDMomentMatch(100,100,0.05,1.0,0.2,m,50000,'LookFloatPut'))
wb.save('FloatPut.xls')


# # END

# In[ ]:




