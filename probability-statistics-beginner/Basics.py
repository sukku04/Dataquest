
# coding: utf-8

# # Analyzing Movie Reviews

# ## 1. Getting to know data

# In[15]:


import pandas as pd
import numpy as np

movies = pd.read_csv("fandango_score_comparison.csv")
movies.head()


# ## 2. Histogram of Fandango and Metacritic

# In[16]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.hist(movies["Metacritic_norm_round"])
plt.show()

plt.hist(movies["Fandango_Stars"])
plt.show()


# Overall, Fandango's review rating is higher than Metacritic. In addition to, the both distributions are biased either Right(Metcritic) or Left(Fandango).
# We need to do randome sampling from of the data and need to do statistic analysis based on the previous exercised to see how clearly different they are.

# ## 3. Mean, Median and Standard deviation

# In[17]:


mean_metacrit = movies["Metacritic_norm_round"].mean()
median_metacrit = np.median(movies["Metacritic_norm_round"])
stdv_metacrit = np.std(movies["Metacritic_norm_round"])
print("Mean of Metacritic_norm_round : ", round(mean_metacrit,2))
print("Median of Metacritic_norm_round : ", median_metacrit)
print("Standard deviation of Metacritic_norm_round : ", round(stdv_metacrit,2))

print("")

mean_fandango = movies["Fandango_Stars"].mean()
median_fandango = np.median(movies["Fandango_Stars"])
stdv_fandango = np.std(movies["Fandango_Stars"])
print("Mean of Fandango_Stars : ", round(mean_fandango,2))
print("Median of Fandango_Stars : ", median_fandango)
print("Standard deviation of Fandango_Stars : ", round(stdv_fandango,2))


# #1. As you can see above histogram chart of Fandango, the chart is right-skewed which means the number of higher rating is contained mainly. That's why mean is higher than median in case of Fandango. 
# Since the mean of large values of data tends to be higher than small values, the data values of Fandago can be considered large.
# 
# #2. The distribution graph of Fandango shows more normalized than Metacritic as above. It reflects that the stdv can be lower as well.
# 
# #3. Since the most of values in Fandago are higher than Metacritic, the mean should be higher as well obviously
# 

# ## 4. Scatter plots

# In[18]:


plt.scatter(movies["Fandango_Stars"],movies["Metacritic_norm_round"])
plt.show()


# In[21]:


movies["fm_diff"] = np.abs(movies["Fandango_Stars"] - movies["Metacritic_norm_round"])


# In[22]:


movies.sort_values(by="fm_diff",ascending=False).head(5)


# ## 5. Correlations (by r-value, linear regression)

# In[39]:


from scipy.stats import pearsonr


r,p_value = pearsonr(movies["Fandango_Stars"],movies["Metacritic_norm_round"])

r


# The meaning of r-value is how related the both values are. 0.18 is considered quite low because the closer to 0, the lesser correlation is.
# We can even see this result from the above scatter plot.

# In[49]:


from scipy.stats import linregress

slope, intercept, rvalue, pvalue, stderr = linregress(movies["Metacritic_norm_round"],movies["Fandango_Stars"])
pred_3 = 3 * slope + intercept

pred_3


# In[50]:


pred_1 = 1 * slope + intercept
pred_1


# In[51]:


pred_5 = 5 * slope + intercept
pred_5


# In[53]:


plt.scatter(movies["Metacritic_norm_round"],movies["Fandango_Stars"])
plt.plot([1.0,5.0],[pred_1,pred_5])
plt.xlim(1,5)
plt.show()


# The linear regression plot refers to the correlation between Metacritic and Fandango and 
