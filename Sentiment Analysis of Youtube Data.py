#!/usr/bin/env python
# coding: utf-8

# # Analysis tasks:
# 
# 1. First we have to perform sentiment analysis on youtube Comments to judge people thinking:
# 2. Second we have to perform EDA for positive comments and analyze its trends
# 3. Third we have to Peform EDA for Negetive comments and also analyze its trends
# 4. We have to analyzing the tags column and identify what are the trending tags on youtube
# 5. Prform Analysis on likes, dislikes and views and will also find how they are corelate to each other
# 6. Perform Emoji Analysis on comments
# 

# In[1]:


# we have to import the necessarry libararies for Data Analysis:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# As this data set is text base so you have to put the error_bad_lines function as False to skip the lines:
comments = pd.read_csv("/Users/ahmad/Data Science BootCamp with Python/Data Analysis Projects\Projects/1-Youtube Text Data Analysis/GBcomments.csv", error_bad_lines = False)


# In[3]:


comments.head()


# In[4]:


# To Analyze the text we have to install Textblob 
get_ipython().system('pip install textblob')


# In[5]:


from textblob import TextBlob


# In[6]:


# Lets text if we perform sentiment on single coloumn from above data set first and analyze its Bihaviour:
TextBlob("It's more accurate to call it the M+ (1000) be...").sentiment.polarity


# In[7]:


comments.isna().sum()


# In[8]:


comments.dropna(inplace = True)


# In[9]:


comments.head()


# In[10]:


# To get all the comments polarity:
polarity = []

for i in comments["comment_text"]:
    polarity.append(TextBlob(i).sentiment.polarity)


# In[11]:


comments["polarity"] = polarity


# In[12]:


comments.head()


# In[13]:


# Know we will see randomllly at what extent we expereience polarity:
comments.head(20)


# ## EDA on Positive Comments:

# In[14]:



Positive_comments = comments[comments["polarity"]==1]


# In[15]:


Positive_comments


# In[16]:


Positive_comments.shape


# In[17]:


# For sake of analyzing the importance of words we have to install the libirary called worldcloud:
get_ipython().system('pip install wordcloud')


# In[18]:


from wordcloud import WordCloud,STOPWORDS


# In[19]:


# we also impoted the stoupword as for the sake of stoping the words whose makes no sence in the line:
stopwords = set(STOPWORDS)


# In[20]:


# As for the analysis we have to join all the data in the form of text in the comments section:
total_comments = ''.join(Positive_comments["comment_text"])


# In[21]:


wordcloud = WordCloud(width = 1000, height = 500, stopwords = stopwords).generate(total_comments)


# In[22]:


wordcloud


# In[23]:


plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')


# ## EDA on Negitive Comments:

# In[24]:


Negitive_comments = comments[comments["polarity"]== -1]


# In[25]:


Negitive_comments


# In[26]:


Negitive_comments.shape


# In[27]:


# As for the analysis we have to again join all the data in the form of text in the comments section:
total_comments = ''.join(Negitive_comments["comment_text"])


# In[28]:


wordcloud = WordCloud(width = 1000, height = 500, stopwords = stopwords).generate(total_comments)


# In[29]:


plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')


# ## Trending Tags

# In[30]:


videos = pd.read_csv("/Users/ahmad/Data Science BootCamp with Python/Data Analysis Projects\Projects/1-Youtube Text Data Analysis/USvideos.csv",error_bad_lines = False)


# In[31]:


videos.head()


# In[32]:


videos.isna().sum()


# In[33]:


tags_vd= ' '.join(videos["tags"])


# In[34]:


tags_vd


# In[35]:


# We have seen in the data there are lots of unorecedent slashes and symbols that would effect our analysis so we should
# remove them for this we will import the libirary of regular expression module:
import re


# In[36]:


tags = re.sub('[^a-zA-z]', ' ', tags_vd)


# In[37]:


tags # Now its only left the text


# In[38]:


# for removing the extra Space we have to repeat the same process:
tags = re.sub(' +' , ' ', tags)


# In[39]:


tags


# In[40]:


wordcloud = WordCloud(width = 1000, height = 500, stopwords = stopwords).generate(tags)


# In[41]:


plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')


# ## Corelation 
# 
# * Lets find out the relation among continuous variables
# * As quite obvious the number of likes have very strong relation with views

# In[43]:


# for the corelation between likes and disllikes we have to use regression plot through seaborn:
sns.regplot(data = videos, x = "views", y = "likes")
plt.title("Regression plot between views and likes")


# we have seen that likes are incressing with incresing the views so these have strong corelation in between

# In[44]:


# same for Dislikes
sns.regplot(data = videos, x = "views", y = "dislikes")
plt.title("Regression plot between views and DiSlikes")


# we have seen that views are incresing then the dislikes are not incresing so these have less corelation in between

# ###### NOw we are going to find the all the components with each other through a better approach :

# In[46]:


df_cor = videos[["likes", "dislikes", "views"]]
df_cor.corr()


# In[47]:


# we will draw a heatmap to show a better version of corelation between all:
sns.heatmap(df_cor.corr(), annot = True)


# ## Emoji Analysis:

# In[48]:


comments.head()


# In[49]:


comments["comment_text"][1]


# In[50]:


# For the emojis there is an special code be assigned to everyone of these for examople:
print('\U0001F600')


# In[51]:


# for the Analysis of emojis we have to import the emojis liberary :
get_ipython().system('pip install emoji')


# In[72]:


import emoji


# In[87]:


get_ipython().system('pip install emoji --upgrade')


# In[88]:


len(comments)


# In[89]:


comment=comments['comment_text'][1]
comment


# In[119]:


[c for c in comment if c in emoji.UNICODE_EMOJI_ENGLISH]


# In[120]:


str=''
for i in comments['comment_text']:
    list=[c for c in comment if c in emoji.UNICODE_EMOJI_FRENCH]
    for ele in list:
        str=str+ele


# In[121]:


len(str)


# In[115]:


str


# ###### As we know we have got the emojis but here for analysis we will have to get all the emojis so:
# * lets create a dictionary of having each emoji with its frequency as well

# In[122]:


result={}
for i in set(str):
    result[i]=str.count(i)


# In[123]:


result


# In[124]:


result.items()


# In[125]:


final={}
for key,value in sorted(result.items(),key =lambda item:item[1]):
    final[key]=value


# In[126]:


final


# In[127]:


## convert dictionary into list for this we have to unzip this dictionary
keys=[*final.keys()]


# In[128]:


values=[*final.values()]


# In[129]:


values


# In[130]:


df=pd.DataFrame({'chars':keys[-20:],'num':values[-20:]})


# In[131]:


df


# In[132]:


import plotly.graph_objs as px
from plotly.offline import iplot


# In[133]:


trace=px.Bar( x=df['chars'], y=df['num'])bb
iplot([trace])


# 
# ### The END 
# * we have analyze all the requirments and also we have performed the  indepth sentiment analysis on through this youtube data.
