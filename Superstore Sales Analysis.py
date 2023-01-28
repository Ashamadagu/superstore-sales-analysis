#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


df = pd.read_csv('superstore_data.csv')
df.head()


# In[6]:


df.describe()


# In[7]:


list(df.columns)


# In[13]:


print("1. What product sold most and why we think it sold most ")
products_df = df.loc[:, ['MntWines',
 'MntFruits',
 'MntMeatProducts',
 'MntFishProducts',
 'MntSweetProducts',
 'MntGoldProds']]

products_df.head()


# In[14]:


products_df.dropna()
products_df = products_df.astype(float)
products_df.head()


# In[15]:


products_df.describe()


# In[16]:


products_df = products_df.apply(pd.to_numeric)


# In[17]:


product_sold_most = products_df.sum().idxmax()
print(product_sold_most + " = product that sold the most")


# In[18]:


print("2. How much was earned from the product that sold most ")
amount_earned = products_df['MntFruits'].sum()
print("Amount earned from the most sold product is {0}".format(amount_earned))


# In[19]:


print("3. What time should we display advertisements to maximize buying of products ")
df


# In[20]:


purchase_df = df.loc[:, ['NumDealsPurchases',
 'NumWebPurchases',
 'NumCatalogPurchases',
 'NumStorePurchases']]
purchase_df


# In[21]:


purchase_df.describe()


# In[22]:


sum_ = purchase_df.sum(axis = 1)
sum_


# In[23]:


purchase_df['total_purchase'] = sum_
purchase_df


# In[24]:


formated_date = pd.to_datetime(df['Dt_Customer']).astype('datetime64[ns]')
formated_date


# In[25]:


# formated_date = formated_date.dt.day
# formated_date


# In[26]:


purchase_df['formated_date'] = formated_date
purchase_df


# In[27]:


total_g_date = purchase_df.loc[:, ['formated_date',
 'total_purchase']]
total_g_date


# In[28]:


total_date_grouped = total_g_date.groupby('formated_date')
totals = total_date_grouped.sum()
totals


# In[29]:


df_totals = pd.DataFrame(totals)
df_totals.iloc[:, 0]


# In[30]:


import matplotlib.pyplot as plt
plt.plot(df_totals.iloc[:, 0])
  
plt.xlabel('Date')
plt.ylabel('Max Purchase')
plt.title('Purchase Arranged By Date')
plt.show()


# In[31]:


maxValues = totals.loc[totals['total_purchase'].idxmax()]
maxValues


# In[32]:


data = totals.rename(columns={'x': 'y'})
data = data.sort_index()
data.head()


# In[33]:


print(f'Number of rows with missing values: {data.isnull().any(axis=1).mean()}')


# In[34]:


steps = 36
data_train = data[:-steps]
data_test  = data[-steps:]


# In[35]:


print(f"Train dates : {data_train.index.min()} --- {data_train.index.max()}  (n={len(data_train)})")
print(f"Test dates  : {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test)})")


# In[36]:


data_test['total_purchase']


# In[37]:


fig, ax=plt.subplots(figsize=(9, 4))
data_train['total_purchase'].plot(ax=ax, label='train')
data_test['total_purchase'].plot(ax=ax, label='test')
ax.legend();


# In[38]:


print("4. Which product has the highest tax amount and freight charges")


# In[39]:


df


# In[40]:


salesOrder = pd.read_excel('DataSet_SalesOrders.xlsx')
salesOrder


# In[41]:


salesOrder.dropna()


# In[42]:


highestTaxAmount = salesOrder.loc[salesOrder['TaxAmt'].idxmax()]
highestTaxAmount


# In[43]:


highestForeignCharges = salesOrder.loc[salesOrder['Freight'].idxmax()]
highestForeignCharges


# In[44]:


list(salesOrder.columns)


# In[45]:


list(df.columns)


# In[46]:


df.head()


# In[47]:


salesOrder.head()


# In[48]:


order = df.loc[:, ['Id','MntWines',
 'MntFruits',
 'MntMeatProducts',
 'MntFishProducts',
 'MntSweetProducts',
 'MntGoldProds']]
order


# In[49]:


orders = 1*(order.set_index('Id') > 0)
orders


# In[50]:


import numpy as np
order_count_relation = pd.DataFrame(data = orders.values.T@orders.values, columns = orders.columns, index = orders.columns)
order_count_relation


# In[51]:


correlations = order_count_relation.corr()
correlations


# In[53]:


print("From the above df, MintWines and MintMeatProducts have a joint value of 0.999764 which is greater than all other values")
print("This indicates that the two are the products that are mostly sold together")


# In[52]:


import seaborn as sns
sns.scatterplot(x="MntWines", y="MntMeatProducts", data=correlations);

