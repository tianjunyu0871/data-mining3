
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist
from math import ceil
get_ipython().magic('matplotlib inline')
df = pd.read_csv("Game/vgsales.csv",encoding="utf-8")


# # 分析数据库信息并进行挖掘

# In[2]:

df.head()


# In[3]:

df.info()
df=df.dropna(axis=0, how='any')
df=df.reset_index(drop=True)#index重排序
df.info()


# In[4]:

##每年发布的游戏和销量##
Year=df.loc[:,'Year'].value_counts()
index=Year.index.values
Year_sum=Year.values
dic1 = {'Year':index, 'game_sum': Year_sum}
Year_frame=pd.DataFrame(dic1)
Year_frame=Year_frame.sort_values(by="Year")
Year_frame=Year_frame.set_index('Year',drop='true')
plt.figure( ) 
Year_frame.plot(kind='bar',figsize=(10,8))
plt.title("Sum of Games Every year")
plt.show()

revenue=df[["Year","Global_Sales"]]
revenue=revenue.groupby(by="Year").sum()
fig = plt.figure( ) 
revenue.plot(kind='bar',figsize=(10,8),color='red')
plt.title("Global_sales(million) Every year")
plt.show()


# **从上面游戏发布数量和销售额随时间变化可以看出，自1997年之后，发行数量激增，伴随而来的是销量的突飞猛进，并在2008年和2009年达到顶峰。
# 最近几年开始下降。 这可能和近年来的其他联网大型游戏的制作有关。**

# In[5]:

#获取平台发售的数量和销售游戏类型相关信息
platGenre = pd.crosstab(df.Platform,df.Genre)
platGenreTotal = platGenre.sum(axis=1).sort_values(ascending = False)
plt.figure(figsize=(8,6))

sns.barplot(y = platGenreTotal.index, x = platGenreTotal.values, orient='h')
plt.show()

platGenreTotal = platGenre.sum(axis=0).sort_values(ascending = False)
plt.figure(figsize=(8,6))

sns.barplot(y = platGenreTotal.index, x = platGenreTotal.values, orient='h')
plt.show()


# **从上面可以看出，这20多年发布游戏最多的平台为DS任天堂的掌上游戏机，索尼PSP系列等，而发布最多的游戏种类前五分别是动作类，运动类，混合类游戏，角色扮演，射击。**
# 
# 
# **最受欢迎的游戏，类型，平台，发行人可以根据销售额的多少来进行反应的，可以从全球，美国，欧洲，日本几个区域类分析**

# In[6]:

area_list=["NA_Sales","EU_Sales","JP_Sales","Global_Sales"]
df_show1=pd.DataFrame(columns=['Year' ,'Rank1','Rank2','Rank3','Rank4'])
df_show2=pd.DataFrame(columns=['Year' ,'Rank1','Rank2','Rank3','Rank4'])
df_show3=pd.DataFrame(columns=['Year' ,'Rank1','Rank2','Rank3','Rank4'])
df_show4=pd.DataFrame(columns=['Year' ,'Rank1','Rank2','Rank3','Rank4'])
df_show=[df_show1,df_show2,df_show3,df_show4]
for i in range(4):
    temp=0
    df_area=df[["Year","Genre","Publisher",area_list[i]]]
    for year in range(int(df_area["Year"].min()+1),int(df_area["Year"].max()-3)):
        df_eachyear = df_area[df_area['Year'] == year]
        df_eachyear=df_eachyear[["Publisher",area_list[i]]].groupby(by="Publisher").count().sort_values(by=area_list[i])[::-1]
        df_array=df_eachyear.head(4).index.values

        df_array= np.append(year,df_array)

        df_show[i].loc[temp]=df_array
        temp+=1


# In[7]:

print("--------------------NA的发行人排名情况(销量前4)---------")

df_show[0].head(int(df_area["Year"].max()-3)-int(df_area["Year"].min()+1))


# In[8]:

print("--------------------EU的发行人排名情况(销量前4)---------")

df_show[1].head(int(df_area["Year"].max()-3)-int(df_area["Year"].min()+1))


# In[9]:

print("--------------------JP的发行人排名情况(销量前4)---------")

df_show[2].head(int(df_area["Year"].max()-3)-int(df_area["Year"].min()+1))


# In[10]:

print("--------------------Global的发行人排名情况(销量前4)---------")

df_show[3].head(int(df_area["Year"].max()-3)-int(df_area["Year"].min()+1))


# **从上面关于发行人前4的排名可以看出，不管是全球，北美，欧洲还是日本，关于游戏发行人的排名基本一样，80，90年代Nintendo任天堂在发行游戏这一块销量非常高，后面逐渐有索尼，EA等大厂加入，10年以后占据优势的是Namco Bandai Games和EA，基本上都是日本和美国的游戏厂商。**
# 
# **上面比较详细的列了4个排名，其他关于类型，平台，游戏的操作类似，就不再重复操作，就只列最受欢迎的一项**
# 
# 
# **下面将对这将对全球近40年的游戏做一个总结的表。**

# In[11]:

df_summary=pd.DataFrame(columns=['Year' ,'most popular game','most popular Genre','most popular platform','most popular publisher'])
    
temp=0
df_all=df[["Year","Name","Platform","Genre","Publisher","Global_Sales"]]
df_names=df_all[["Year","Name","Global_Sales"]]
df_Genre=df_all[["Year","Genre","Global_Sales"]]
df_Publisher=df_all[["Year","Publisher","Global_Sales"]]
df_Platform=df_all[["Year","Platform","Global_Sales"]]
for year in range(int(df_all["Year"].min()+1),int(df_all["Year"].max()-3)):
    ##先处理游戏名
    df_eachyear = df_names[df_names['Year'] == year]
    df_eachyear1=df_eachyear[["Name","Global_Sales"]].groupby(by="Name").count().sort_values(by="Global_Sales")[::-1]
    df_names1=df_eachyear1.head(1).index.values

    ##处理游戏类型
    df_eachyear = df_Genre[df_Genre['Year'] == year]
    df_eachyear=df_eachyear[["Genre","Global_Sales"]].groupby(by="Genre").count().sort_values(by="Global_Sales")[::-1]
    df_Genre1=df_eachyear.head(1).index.values
                            
    ##处理发行商
    df_eachyear = df_Publisher[df_Publisher['Year'] == year]
    df_eachyear=df_eachyear[["Publisher","Global_Sales"]].groupby(by="Publisher").count().sort_values(by="Global_Sales")[::-1]
    df_Publisher1=df_eachyear.head(1).index.values                   
    ##处理游戏平台
    df_eachyear = df_Platform[df_Platform['Year'] == year]
    df_eachyear=df_eachyear[["Platform","Global_Sales"]].groupby(by="Platform").count().sort_values(by="Global_Sales")[::-1]
    df_Platform1=df_eachyear.head(1).index.values
    
    
                            
    df_array= np.append(year,df_names1)
    df_array=np.append(df_array,df_Genre1)
    df_array=np.append(df_array,df_Platform1)
    df_array=np.append(df_array,df_Publisher1)
    df_summary.loc[temp]=df_array
    temp+=1
df_summary.head(int(df_all["Year"].max()-3)-int(df_all["Year"].min()+1))


# **上表反应了充1981年~2016年之内各年最受欢迎的游戏，游戏类型，游戏平台，游戏厂商的信息，各列相互独立，可以看出80~90年代主流为运动游戏，2000年之后动作类游戏已逐渐成为主流，同时Electronic Arts的PS游戏平台也有越来越多的玩家开始关注。**

# # 每年电子游戏销售预测

# **从上面那个销量图可以看出销量从少到多再到少，大体上符合一个高斯分布的模型，虽然有悖常理（更多可能是数据库信息更新的原因），但符合库的性质，因此可以对其进行函数拟合来预测每一年的销量**

# In[83]:

####多项式拟合############
train_x=revenue.head(36).index.values.flatten()
#标签labes
train_y=revenue.head(36).values.flatten()
f1 = np.polyfit(train_x, train_y, 5)
print('拟合的参数 is :\n',f1)
p1 = np.poly1d(f1)
print('多项式为 :\n',p1)
yvals = p1(train_x)
years=[]
for i in range(1980,2016):
    years.append(i)
plot_frame=pd.DataFrame(columns=['Year' ,'predicted_sales','real_sales','difference'])

for i in range(0,36):
    temp_array=[]
    temp_array=np.append(temp_array,years[i])
    temp_array=np.append(temp_array,yvals[i])
    temp_array=np.append(temp_array,train_y[i])
    temp_array=np.append(temp_array,round(np.abs(train_y[i]-yvals[i]),2))
    plot_frame.loc[i]=temp_array
    

plot1 = plt.plot(train_x, train_y, 's',label='original values')
plot2 = plt.plot(train_x, yvals, 'r',label='polyfit values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4) #指定legend的位置右下角
plt.title('polyfitting')
plt.show()


# **上述为多项式拟合，拟合的多项式公式如上，同时附上拟合的曲线图，下面为每年预测的结果和真实值之间的对比**

# In[84]:

print("每年通过三项式预测的结果为(1980~2015):\n")
plot_frame.head(36)


# In[85]:

########高斯拟合###############
from scipy.optimize import curve_fit  
from scipy import asarray as ar,exp

x = train_x-1980
y = train_y
   
def gaussian(x,*param):  
    return param[0]*np.exp(-np.power(x - param[2], 2.) / (2 * np.power(param[4], 2.)))+param[1]*np.exp(-np.power(x - param[3], 2.) / (2 * np.power(param[5], 2.)))  
   
   
popt,pcov = curve_fit(gaussian,x,y,p0=[3,4,3,6,1,1])  #p0六个参数分别对应上面*param的六个参数，分别对应两个高斯函数求和。

print("高斯拟合的6个参数为:"+str(popt))
print("\n拟合的曲线如下:")

plt.plot(train_x,y,'b+:',label='data')  
plt.plot(train_x,gaussian(x,*popt),'ro:',label='fit')  
plt.legend()  
plt.show()


# **从感觉上来看高斯拟合的结果要比多项式拟合的结果效果要好,下面为每年的结果和真实值之间的比较**

# In[87]:

print("每年通过高斯拟合预测的结果为(1980~2015):\n")
gauss_frame=pd.DataFrame(columns=['Year' ,'predicted_sales','real_sales','difference'])
gauss_result=gaussian(x,*popt)
for i in range(0,36):
    temp_array2=[]
    temp_array2=np.append(temp_array2,years[i])
    temp_array2=np.append(temp_array2,gauss_result[i])
    temp_array2=np.append(temp_array2,train_y[i])
    temp_array2=np.append(temp_array2,round(np.abs(train_y[i]-gauss_result[i]),2))
    gauss_frame.loc[i]=temp_array2
gauss_frame.head(36)


# In[97]:

#使用SVR进行回归测试，将数据以0.8分为训练数据和测试数据

train_x=revenue.head(36).index.values
#标签labes
train_y=revenue.head(36).values
#拆分数据集，建立训练数据集和测试数据集
train_y=train_y.flatten()
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
x_train , x_test , y_train , y_test = train_test_split(train_x ,
                                                      train_y ,
                                                      train_size = .8)
#将测试数据特征转换成二维数组行数*1列
train_x=train_x.reshape(-1,1)
train_y=train_y.reshape(-1,1)
x_train=x_train.reshape(-1, 1)
y_train=y_train.reshape(-1, 1)
x_test=x_test.reshape(-1, 1)
y_test=y_test.reshape(-1, 1)


model = svm.SVR(kernel='linear')
#fit函数训练模型
model.fit(x_train , y_train)

result=model.predict(x_test)

result_all=model.predict(train_x)
mse=mean_squared_error(result,y_test)

train_y=train_y.flatten()
print("测试集上的MSE="+str(mse))

svr_frame=pd.DataFrame(columns=['Year' ,'predicted_sales','real_sales','difference'])
for i in range(0,36):
    temp_array3=[]
    temp_array3=np.append(temp_array3,years[i])
    temp_array3=np.append(temp_array3,result_all[i])
    temp_array3=np.append(temp_array3,train_y[i])
    temp_array3=np.append(temp_array3,round(np.abs(train_y[i]-result_all[i]),2))
    svr_frame.loc[i]=temp_array3
svr_frame.head(36)


# **从上面可以看出用SVR的效果不好，在测试集上MSE非常大，拟合效果不佳，其原因是作为回归的特征数量不够，这里只是一维，要想效果好，需要多提取写特征**
