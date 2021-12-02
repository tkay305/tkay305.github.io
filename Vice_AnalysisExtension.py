#!/usr/bin/env python
# coding: utf-8

# # **Exploratory Analysis of Vice Social Media Data: 2018 to May 2021**
# 

# **1. Loading In Necessary Packages**:
# 

# In[14]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import dates as mpl_dates
import re
import nltk
import textblob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
get_ipython().system('pip install wordcloud --user')
from wordcloud import WordCloud
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error
from math import sqrt

get_ipython().run_line_magic('matplotlib', 'inline')


# **2. Reading in dataset and checking it's properties:**
# 

# In[15]:




df=pd.read_csv(r"C:\Users\26097\Documents\vice.csv")
print(df.head())
print(df.columns)
df.shape
df.describe()


# In[219]:


df.nunique()


# In[49]:


pd.set_option('display.float_format', lambda x: '%.5f' % x)
df_core[['Total_Interactions','Total_Views', 'Angry', 'Love', 'Care', 'Haha', 'Wow', 'Sad']].describe()


# In[16]:


for column in df:
    unique_vals=np.unique(df[column])
    num_values=len(unique_vals)
    if num_values < 10:
        print('The number of values for feature {} : {} --{}'.format(column, num_values, unique_vals))
    else:
        print('The number values for feature {} : {}'.format(column, num_values))


# In[17]:


df.isnull().sum()


# **4. Edits to Numerical Columns**

# In[18]:


df['Message'] = df['Message'].fillna('').apply(str)


# In[19]:


print(df['Total Interactions'])


# In[21]:


df['Total Interactions'] = pd.to_numeric(df['Total Interactions'].str.replace(',',''), errors='coerce')


# In[22]:


print(df['Total Interactions'])


# **5.Drop Rows that won't be used in analysis**

# In[23]:



df_core=df.drop(['Facebook Id','Page Created','Page Category','Page Admin Top Country','Page Description', 'Likes at Posting', 'Followers at Posting','URL', 'Link', 'Final Link', 'Image Text', 'Link Text', 'Description', 'Sponsor Id', 'Sponsor Name', 'Sponsor Category'], axis=1)
df_core = df_core.rename(columns={'Total Views': 'Total_Views', 'Total Views For All Crossposts': 'Crossposts_Total_Views', 'Total Interactions': 'Total_Interactions'})


# **6. Create Time Based Measures**

# In[24]:



df_core['Post Created']=pd.to_datetime(df_core['Post Created'],yearfirst=True, utc=True)
df_core['Year']=pd.DatetimeIndex(df_core['Post Created']).year
df_core['Month']=pd.DatetimeIndex(df_core['Post Created']).month
df_core['Weekday']=df_core['Post Created'].dt.day_name()
df_core['Time']=df_core['Post Created'].dt.time
df_core.drop(df_core.loc[df_core['Video Length']== "336:00:00"].index, inplace=True)
df_core['Duration']=pd.to_datetime(df_core['Video Length'],format="%H:%M:%S").dt.time
def hr_func(Time):
    return Time.hour

df_core['Hour_Posted'] = df_core['Time'].apply(hr_func)

def min_func(Time):
    return Time.minute

df_core['Video_Minutes'] = df_core['Duration'].apply(min_func)


df_core.head()


# **7. View Relationships among variables of interest**

# Across the entire dataset, median user views per post sit at about 45 000, while average views are at 288 000. For user interactions, the median value is 312 while the mean is 3230. There is a lot of variability in the two data pointa, evident in the spread  of the data points from the centre of the graph.
# 

# In[25]:


ang=df_core.filter(['Angry','Post Views', 'Total_Views','Total_Interactions', 'Crossposts_Total_Views','Likes', 'Shares', 'Comments'])
ang_corr=ang.corr()
anger_matrix= sns.heatmap(ang_corr, xticklabels=ang_corr.columns, yticklabels=ang_corr.columns, annot=True)


# The Angry reaction variable has low correlations with Views, and User Engagement metrics. This suggests that videos likely to make Vice users react with an angry emoticon are unlikely to result in high post views, interactions, likes or comments. Posts associated with angry reactions also have the lowest correlation with post views. 

# In[155]:


anger_plot=sns.relplot(x="Total_Interactions", y="Total_Views",hue="Angry",size= "Shares", height=5, aspect=3, facet_kws=dict(sharex=False),
             legend="auto", data=df_core)
anger_plot.fig.suptitle("User Views and Interactions for angry Reactions")
anger_plot.set(ylim=(0, 8000000))
anger_plot.set(xlim=(0, 400000))
leg = anger_plot._legend
leg.set_bbox_to_anchor([1,0.5])
plt.ticklabel_format(style='plain', axis='both',useOffset=False)
plt.savefig("User Views and Interactions for angry Reactions.jpg")


# For video posts that has users reacting with angry emoticons, user views are concentrated between 0 and 1M, though there is some of spread in the data points. Interactions range mostly between 0 and 50 000 per post, with a few outliers. The share variable has the highest correlation with angry reactions, and the distribution of shared posts looks to be under 80 000 with a few exceptions.

# In[257]:


sad=df_core.filter(['Sad','Post Views', 'Total_Views','Total_Interactions', 'Crossposts_Total_Views' 'Likes', 'Shares', 'Comments'])
sad_corr=sad.corr()
sad_matrix= sns.heatmap(sad_corr, xticklabels=sad_corr.columns, yticklabels=sad_corr.columns, annot=True)


# The sad reaction variable has low correlations with Views, and User Engagement metrics, but higher than those of the angry reactions.

# In[68]:


sad_plot=sns.relplot(x="Total_Interactions", y="Total_Views",hue="Sad",size= "Shares", height=5, aspect=3, facet_kws=dict(sharex=False),
             legend="auto", data=df_core)
sad_plot.fig.suptitle("User Views and Interactions for sad reactions")
sad_plot.set(ylim=(0, 8000000))
sad_plot.set(xlim=(0, 250000))
leg = sad_plot._legend
leg.set_bbox_to_anchor([1,0.5])
plt.ticklabel_format(style='plain', axis='both',useOffset=False)


# Video posts that have users reacting with sad emoticons have user views mostly between 0 and 1M, though there is a lot of spread in the total views data points. Interactions range mostly between 0 and 50 000 per post, with a few outliers.The sad emoticon is used to reaction to more video posts than angry, with the lower bound for this variable being between 0 and 30,000 reactions per post. The median reaction of 1 is the same as that of angry, but there is less variability in the sad reactions. The share variable has the highest correlation with sad reactions, and the general distribution of shared posts is under 80 000.

# In[259]:


love=df_core.filter(['Love','Post Views', 'Total_Views','Total_Interactions', 'Crossposts_Total_Views' 'Likes', 'Shares', 'Comments'])
love_corr=love.corr()
love_matrix= sns.heatmap(love_corr, xticklabels=love_corr.columns, yticklabels=love_corr.columns, annot=True)


# In[70]:


love_plot=sns.relplot(x="Total_Interactions", y="Total_Views",hue="Love",size= "Shares", height=5, aspect=3, facet_kws=dict(sharex=False),
             legend="auto", data=df_core)
love_plot.fig.suptitle("User Views and Interactions for love reactions")
love_plot.set(ylim=(0, 8000000))
love_plot.set(xlim=(0, 200000))
leg = love_plot._legend
leg.set_bbox_to_anchor([1,0.5])
plt.ticklabel_format(style='plain', axis='both',useOffset=False)


# The data points for both views and interactions where users gave a love reaction have more spread than that of sad and angry, despite the concentration of user views and interactions being within the same range. This means there is an increased probablity of users viewing and interacting with love content. The range of reactions mostly sits between 0-20 000, with little variation. 

# In[261]:


haha=df_core.filter(['Haha','Post Views', 'Total_Views','Total_Interactions', 'Crossposts_Total_Views' 'Likes', 'Shares', 'Comments'])
haha_corr=haha.corr()
haha_matrix= sns.heatmap(haha_corr, xticklabels=haha_corr.columns, yticklabels=haha_corr.columns, annot=True)


# In[64]:


haha_plot=sns.relplot(x="Total_Interactions", y="Total_Views",hue="Haha",size= "Shares", height=5, aspect=3, facet_kws=dict(sharex=False),
             legend="auto", data=df_core)
haha_plot.fig.suptitle("User Views and Interactions for haha reactions")
haha_plot.set(ylim=(0, 10000000))
haha_plot.set(xlim=(0, 400000))
leg = haha_plot._legend
leg.set_bbox_to_anchor([1,0.5])
plt.ticklabel_format(style='plain', axis='both',useOffset=False)
plt.savefig("User Views and Interactions for haha reactions.jpg")


# Posts that have users reacting with haha emoticon have a range between 0-80 000 reactions per post. This is the highest interaction per post, based on the user reaction measure. Haha reactions are highly correlated with getting a comment interaction. Haha reactions have the highest correlation with user engagement. This suggests posting video content that is funny is the most likely to get views, interacted with, shared and commented on. 

# In[263]:


care=df_core.filter(['Care','Post Views', 'Total_Views','Total_Interactions', 'Crossposts_Total_Views' 'Likes', 'Shares', 'Comments'])
care_corr=care.corr()
care_matrix= sns.heatmap(care_corr, xticklabels=care_corr.columns, yticklabels=care_corr.columns, annot=True)


# In[74]:


care_plot=sns.relplot(x="Total_Interactions", y="Total_Views",hue="Care",size= "Shares", height=5, aspect=3, facet_kws=dict(sharex=False),
             legend="auto", data=df_core)
care_plot.fig.suptitle("User Views and Interactions for care reactions")
care_plot.set(ylim=(0, 8000000))
care_plot.set(xlim=(0, 125000))
leg = care_plot._legend
leg.set_bbox_to_anchor([1,0.5])
plt.ticklabel_format(style='plain', axis='both',useOffset=False)


# In[265]:


wow=df_core.filter(['Wow','Post Views', 'Total_Views','Total_Interactions', 'Crossposts_Total_Views' 'Likes', 'Shares', 'Comments'])
wow_corr=wow.corr()
wow_matrix= sns.heatmap(wow_corr, xticklabels=wow_corr.columns, yticklabels=wow_corr.columns, annot=True)


# In[75]:


wow_plot=sns.relplot(x="Total_Interactions", y="Total_Views",hue="Wow",size= "Shares", height=5, aspect=3, facet_kws=dict(sharex=False),
             legend="auto", data=df_core)
wow_plot.fig.suptitle("User Views and Interactions for care reactions")
wow_plot.set(ylim=(0, 8000000))
wow_plot.set(xlim=(0, 125000))
leg = wow_plot._legend
leg.set_bbox_to_anchor([1,0.5])
plt.ticklabel_format(style='plain', axis='both',useOffset=False)


# **8. Investigate how variables move over time**

# In[141]:


user_interaction= sns.catplot(x="Year", y="Total_Interactions", hue = "User Name", kind="bar", height=5, aspect=3, palette="ch:.5", edgecolor=".6",data=df_core)
user_interaction.fig.suptitle("User Interactions by Year and User Name")
leg = user_interaction._legend
leg.set_bbox_to_anchor([1,0.5])
plt.savefig("User Interactions by Year.jpg")


# In[106]:


posting_time= sns.catplot(x="Year", y="Hour_Posted" , hue= "User Name", kind="bar", height=5, aspect=3, palette="ch:.25", edgecolor=".6",data=df_core)
posting_time.fig.suptitle("Average Posting Time by Year and User Name")
leg = posting_time._legend
leg.set_bbox_to_anchor([1,0.5])


# In[142]:


posting_time_m = sns.catplot(y= "Hour_Posted", x=  "Month" , hue= "User Name", kind="bar",height=5, aspect=3, palette="ch:.25", edgecolor=".6",data=df_core)
posting_time_m.fig.suptitle("Average Posting Time by Month and User Name")
leg = posting_time_m._legend
leg.set_bbox_to_anchor([1,0.5])
df_core.sort_values('Month', inplace=True)
plt.savefig("Average Posting Time User Name.jpg")


# In[145]:


video_length=sns.catplot(x="Year", y="Video_Minutes" , hue= "User Name" , kind="bar", height=5, aspect=3, palette="ch:.25", edgecolor=".6",data=df_core)
video_length.fig.suptitle("Video Duration by Year and User Name")
leg = video_length._legend
leg.set_bbox_to_anchor([1,0.5])
plt.savefig("Video Duration by Year and User Name.jpg")


# In[114]:


video_length_m=sns.catplot(y="Video_Minutes" , x="Month" , hue="User Name"  , kind="bar", height=5, aspect=3, palette="ch:.25", edgecolor=".6",data=df_core)
video_length_m.fig.suptitle("Video Duration by Month and User Name")
leg = video_length_m._legend
leg.set_bbox_to_anchor([1,0.5])
df_core.sort_values('Month', inplace=True)


# In[144]:


posting_day= sns.catplot(x="Total_Views", y="User Name" , hue= "Weekday", kind="bar", height=5, aspect=3, palette="ch:.25", edgecolor=".6",data=df_core)
posting_day.fig.suptitle("Total_Views by Weekday and User Name")
leg = posting_day._legend
leg.set_bbox_to_anchor([1,0.5])
plt.savefig("Total_Views by Weekday and User Name.jpg")


# #Summary of Findings of Time Based Data:
# - viceuk has grown to have the highest user interactions, averaging 15,000 in 2021.
# - Each user name has a consistent strategy around peak time to make video posts over the years
# - viceuk experiences "peak days" - days when posts have higher views. In order, this is Tuesday, Thursday and Saturday (with some variation in the views). Do they release original content on those days?
# - With the exception of vicetv, every user name has increased the average duration of the video content that they post over the years.
# 

# **9 Analyzing User Sentiment**

# In[132]:


df_rxn=df_core.filter(['Love', 'Wow', 'Haha', 'Sad', 'Angry', 'Care', 'Year', 'User Name', ], axis=1)
df_rxn.head()


# In[133]:


df_rxn = pd.melt(df_rxn, id_vars = ['Year','User Name'], value_vars = ['Love', 'Wow', 'Haha', 'Sad', 'Angry', 'Care'], var_name = 'Reaction', value_name = 'Reaction_Count')
df_rxn.head()


# In[118]:


user_reactions=sns.catplot(x="Year", y="Reaction_Count", hue="Reaction", kind="bar",palette="ch:.25", height=5, aspect=3, edgecolor=".6", data=df_rxn)
user_reactions.fig.suptitle("User Reactions by Year")
leg = user_reactions._legend
leg.set_bbox_to_anchor([1,0.5])


# In[96]:


df_msg= df_core.filter(['Year','Message','Total_Views', 'Total_Interactions', 'Angry', 'Sad', 'Love', 'Haha', 'Wow', 'Care', 'User Name'], axis=1)


# In[97]:


# Define a function to clean the text
def clean(text):
# Removes all special characters and numericals leaving the alphabets
    text = re.sub('[^A-Za-z]+', ' ', text)
    return text

# Cleaning the text in the review column
df_msg['Clean_Text'] = df_msg['Message'].apply(clean)
df_msg.head()


# In[98]:



from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords

from nltk.corpus import wordnet

# POS tagger dictionary
pos_dict = {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}
def token_stop_pos(text):
    tags = pos_tag(word_tokenize(text))
    newlist = []
    for word, tag in tags:
        if word.lower() not in set(stopwords.words('english')):
            newlist.append(tuple([word, pos_dict.get(tag[0])]))
    return newlist

df_msg['POS tagged'] = df_msg['Clean_Text'].apply(token_stop_pos)
df_msg.head()


# In[99]:


from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
def lemmatize(pos_data):
    lemma_rew = " "
    for word, pos in pos_data:
        if not pos:
            lemma = word
            lemma_rew = lemma_rew + " " + lemma
        else:
            lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
            lemma_rew = lemma_rew + " " + lemma
    return lemma_rew

df_msg['Lemma'] = df_msg['POS tagged'].apply(lemmatize)
df_msg.head()


# In[100]:


from textblob import TextBlob
# function to calculate subjectivity
def getSubjectivity(Clean_Text):
    return TextBlob(Clean_Text).sentiment.subjectivity
    # function to calculate polarity
def getPolarity(Clean_Text):
    return TextBlob(Clean_Text).sentiment.polarity

# function to analyze the reviews
def analysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'
    
df_msg['Subjectivity'] = df_msg['Lemma'].apply(getSubjectivity) 
df_msg['Polarity'] = df_msg['Lemma'].apply(getPolarity) 
df_msg['Analysis'] = df_msg['Polarity'].apply(analysis)
df_msg.head()
    


# In[129]:


analysis=sns.catplot(x="Analysis", y="Total_Views", height=5, aspect=3, palette="ch:.25", edgecolor=".6",data=df_msg)
analysis.fig.suptitle("User Views against Message Analysis")
leg = analysis._legend


# In[123]:


def get_top_n_words(corpus, n=None):
    vec = sk.feature_extraction.text.CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(df_msg['Clean_Text'], 30)
for word, freq in common_words:
    print(word, freq)
df2 = pd.DataFrame(common_words, columns = ['Clean_Text' , 'count'])


# In[156]:


df2['Clean_Text']=df2['Clean_Text'].apply(str)
polarity =sns.relplot(x="Polarity", y="Total_Views", hue="Analysis", palette="ch:.25", height=5, aspect=3, edgecolor=".6",data=df_msg)
polarity.fig.suptitle("Views against Message Polarity")
leg = polarity._legend
plt.savefig("Views against Message Polarity.jpg")


# In[137]:


df_msg['top_words'] = df_msg.loc[df_msg['Total_Interactions'] > 300000, 'Clean_Text']


# In[147]:



top_words=sns.catplot(x="Total_Views", y="top_words", hue="Analysis", kind="bar" ,palette="ch:.25", edgecolor=".6",data=df_msg)
top_words.fig.suptitle("Top Words against Message Polarity")
plt.savefig("Words against Message Polarity.jpg")


# In[149]:


top_words2=sns.catplot(x="Total_Interactions", y="top_words", hue= "Analysis", kind="bar" ,palette="ch:.25", edgecolor=".6",data=df_msg)

top_words2.fig.suptitle("Top Words against Message Polarity")


# In[151]:


from wordcloud import WordCloud, STOPWORDS


# In[275]:



comment_words = ''
stopwords = set(STOPWORDS)


for val in df2.Clean_Text:
     
    # typecaste each val to string
    val = str(val)
 
    # split the value
    tokens = val.split()
     
    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
     
    comment_words += " ".join(tokens)+" "
    

wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)
 
# plot the WordCloud image                      
word_cloud=plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()
plt.savefig("Popular Words.jpg")


# **User sentiment findings:**
# - Video Posts with low positive polarity in the text have some variation in total views.Overall, it appears that videos with text that is highly positive or highly nea
# - Over time, user reactions to funny content have increased. Posts tagged positive also received the highest views
# - Chainz content appears to get the highest views and interactions
# 

# **10. Extension to Exploration: Modelling Views in relation to user reaction**

# In[207]:


from sklearn import linear_model


# In[255]:


explanators=['Love','Wow','Haha' ,'Sad','Angry', 'Care']
response=['Total_Views']

X=df_core[explanators].values.reshape(-1, len(explanators))
y=df_core[response].values


# In[256]:


print(X.shape)
print(y.shape)


# In[257]:


reg=linear_model.LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)


# **Linear Model Results**:

# In[258]:


print("Coefficients: \n", reg.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("R-squared: %.2f" % r2_score(y_test, y_pred))
print("Intercept: \n" , reg.intercept_)


# The linear model that attempts to explain for user views using user reactions estimates that each post will get a base viewership of about 166 000 views per post. Each reaction variable is estimated to positively impact views, with the coefficient on the 'Care' variable estimated to have the largest positive impact on views. This model has a high mean square error and relatively low R-square - predictions based off only user reactions are not reliable, but the key takeaway is that each reaction variable affects views positively, that is... Any kind of post, positive or negative is likely to contribute to total views.

# **Model Predictions**

# In[265]:


df_core.iloc[3147,:]


# In[263]:


X_pred = np.array([94, 23, 266, 8, 7,7])
X_pred = X_pred.reshape(-1, len(explanators))
reg.predict(X_pred)


# After checking a random line of data, the prediction power of the regression model is put to the test. In this case, it underestimates user reactions by over 180 000 Views - User reactions alone are not a very good indicator of how a video post will perform. 

# **Model Residuals**

# In[253]:


plt.hist(residuals)
plt.ticklabel_format(style='plain', axis='both',useOffset=False)


# Model residuals are on a negative scale, and peak somewhere in the low hundred thousands. We would therefore expect the model to generally overpredict total views.

# **Predicted against actual Views**

# In[252]:


sns.regplot(x=y_test, y=y_pred, ci=None, color="b")
plt.ticklabel_format(style='plain', axis='both',useOffset=False)


# The plot above shows that the model has generally better predictor power for total views below 5M. Beyond that there is huge variability in the predictions.

# **Next Steps**
# 
# At this stage, I would run a regression based of the sentiment analysis portion of the data set. The relationships of interest would be a numerical representation of the "Postive, Negative, Neutral" scale, against the dependent variable of Total Views. Unfortunately, I run out of time and was not able to complete that portion of analysis. Running linear regressions on subsets of the data was a prelude to variable selectionThis would have allowed the ability to describe which variables are most important  to determining user views, and then user interaction. 
# 

# **Final Words**
# 
# This was my first time working with a social media dataset, which I did not realise would be as much of a challenge as it was. I've really enjoyed the opportunity to navigate around how social media sentiment analysis could work. I appreciate the opportunity to have worked with this data. My curiosity will probably have me attempting to finish off building out this model.
# 

# In[ ]:




