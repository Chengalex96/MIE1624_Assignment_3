# MIE1624_Assignment_3 - NLP and Sentiment Analysis

I am working on two datasets: Canadian Elections 2021 and sentiment analysis based on Twitter tweets.
 
Want to answer the question: What can public opinion on Twitter tell us about the Canadian political landscape in 2021?

The goal is to use sentiment analysis on Twitter data to get insight into the Canadian Elections. For this assignment, we've pulled tweets regarding the Canadian elections from the announcement of the 2021 election to the day before the election for your analysis.

**Part 1: Data Analysis**

Convert both CSV files to a pandas dataframe:
- Canadian election has 10002 rows and 3 columns (text, sentiment, and negative reason)
- Sentiment analysis has 550,391 rows and 3 columns (Tweet ID, text, label with a 1 being positive

**All html tags and attributes (i.e., /<[^>]+>/) are removed:**
df_clean['text'] = df['text'].str.replace('<[^<]+?>', '')

**Html character codes (i.e., &...;) are replaced with an ASCII equivalent**: import HTML

df_clean['text'] = html.unescape(df_clean['text'])

**All URLs are removed.** - import re

df_clean['text'] = df_clean['text'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])

**Punctuations are removed**: 

df_clean['text'] = df_clean['text'].str.replace('[^\w\s]', '')

**All characters in the text are in lowercase.**:

df_clean['text'] = df_clean['text'].apply(lambda x: ' '.join(x.lower() for x in x.split()))

df_clean['negative_reason'] = df_clean['negative_reason'].fillna("nan")

**All stop words are removed. Be clear in what you consider as a stop word.**:

stop words from nltk: import nltk, nltk.download('stopwords'), from nltk.corpus import stopwords, stop = stopwords.words('english')

df_clean['text'] = df_clean['text'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))

df_clean['negative_reason'] = df_clean['negative_reason'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))

**Pat 2: Exploratory Analysis**

Determine the political party of a tweet and unknown if undefined:

- df_clean['party'] = 'unknown'
- df_clean.loc[(df_clean.text.str.contains('thejagmeetsingh') == True),'party']='ndp'
- df_clean.loc[(df_clean.text.str.contains('ndp') == True),'party']='ndp'
- df_clean.loc[(df_clean.text.str.contains('peoples party') == True),'party']='ppc'
- df_clean.loc[(df_clean.text.str.contains('ppc') == True),'party']='ppc'
- df_clean.loc[(df_clean.text.str.contains('bernier') == True),'party']='ppc'
- df_clean.loc[(df_clean.text.str.contains('conservative') == True),'party']='conservatives'
- df_clean.loc[(df_clean.text.str.contains('otoole') == True),'party']='conservatives'
- df_clean.loc[(df_clean.text.str.contains('liberal') == True),'party']='liberal'
- df_clean.loc[(df_clean.text.str.contains('trudeau') == True),'party']='liberal'

This is somewhat skewed if it has multiple parties mentioned in the tweet, parties at the bottom will take the label, this makes more sense since they are more popular

Count:
party
liberal          475
unknown          251
conservatives    208
ppc               48
ndp               20

![image](https://github.com/Chengalex96/MIE1624_Assignment_3/assets/81919159/1f63e0d7-2c90-4de0-af79-d18331c81a6b)

Number of positive vs negative sentiments labels:

![image](https://github.com/Chengalex96/MIE1624_Assignment_3/assets/81919159/d6b52045-9176-4db3-88c0-ffe9cde07847)

**Model 3: Model Preparation**

Check if there's a trend between the length of the tweet and the sentiment - no trend

Find the most common words that are located in both datasets, create a train/test model, and only keep the 125 most common words.

**Bag of Words**: Creates separate columns for each word and counts the number of times they are used for each row (385k)
- Using CountVectorizer(), fit_transform(X_train)

- Must standardize the data as well

**Term Frequency - Inverse Document Frequency (TD-IDF)**
- Using TfidfVectorizer(), fit_transform(X_train)

**Logistic Regression Bag of Words**:

model_lr_bow = LogisticRegression()    

model_lr_bow.fit(X_train_bow, y_train)

Logistic Regression bag of words got an accuracy of 81.97% on the testing set

**Logistic regression tfidf**:

model_lr_tfidf = LogisticRegression()    

model_lr_tfidf.fit(X_train_tfidf, y_train)

Logistic Regression tfidf got an accuracy of 81.93% on the testing set

**k-NN Bag of Words**:

model_knn_bow = KNeighborsClassifier(n_neighbors=3)

model_knn_bow.fit(X_train_bow,y_train)

KNN Bag of Words got an accuracy of 80.0% on the testing set

**k-NN tfidf**:

model_knn_tfidf = KNeighborsClassifier(n_neighbors=3)

model_knn_tfidf.fit(X_train_tfidf,y_train)

predictions_knn_tfidf= model_knn_tfidf.predict(X_test_tfidf)

K-NN tfidf got an accuracy of 79.62% on the testing set

**Naive Bayes Bag of Words**:

model_gnb_bow = GaussianNB()

model_gnb_bow.fit(X_train_bow,y_train)

predictions_gnb_bow= model_gnb_bow.predict(X_test_bow)

Naive Bayes bag of words got an accuracy of 67.48% on the testing set

**Naive Bayes tfidf**:

model_gnb_tfidf = GaussianNB()

model_gnb_tfidf.fit(X_train_tfidf,y_train)

predictions_gnb_tfidf= model_gnb_tfidf.predict(X_test_tfidf)

Naive Bayes tfidf got an accuracy of 67.47% on the testing set

SVM Bag of Words: SVM Bag of Words got an accuracy of 82.46% on the testing set

SVM tfidf: SVM tfidf got an accuracy of 82.64% on the testing set

Decision tree bag of words: Decision tree bag of words got an accuracy of 82.25% on the testing set

Decision tree tfidf: Decision tree tfidf got an accuracy of 81.98% on the testing set

Random forest bag of words: Random forest bag of words got an accuracy of 81.96% on the testing set

Random forest tfidf: Random forest tfidf got an accuracy of 81.95% on the testing set

XGBoost bag of words: XGBoost bag of words got an accuracy of 82.03% on the testing set

XGBoost tfidf: XGBoost tfidf got an accuracy of 81.57% on the testing set

Using minimal features, the Decision tree bag of words gave the best accuracy with 83.42%, however, the process will take much longer or crash if I decide to use more features. However, accuracy increases as I increase the number of features. (Using 500 features for KNN increased the accuracy to around 85%). 

Based on training different models of the sentiment analysis dataset, the model can predict the correct sentiment with an accuracy of 83% which is pretty decent.  

![image](https://github.com/Chengalex96/MIE1624_Assignment_3/assets/81919159/f3a38b53-b2f2-4ded-aa6d-9d2cf6bf2c98)

**Model 4: Model Implementation and Tuning**

Prepare the Bag of Words for the Election Data

Standardize Data

Add a label that puts negative sentiment as 0 and positive as 1

Decision tree bag of words - a model that gave the highest accuracy for sentiment analysis

Random forest tfidf

The model only matches the sentiment labeled in the Canadian elections data about 64% of the time which isn't as good. One reason may be that people tweet/express emotions differently on Twitter when it comes to talking about politics and the elections. Another reason is that the words used in the election dataset may differ a lot and may be used in a different context (it. sarcasm). 

**Metrics used to evaluate models:**

- One method to improve accuracy is to include low-frequency words, the default setting of min_df is 1, and no words are ignored. However, due to insufficient memory, the top frequent words are chosen and limited to make it feasible to run the code in a few minutes compared to a few hours. 
- Another method is to perform stemming or lemmatization, this will allow us to obtain/ combine the meaning of similar words. We can also add more stop words to get a more representative list of words that have meaning. 

**Visualization of the sentiment prediction results and the true sentiment for each of the 4 parties - Confusion matrix**:

The best model is tfidf with random forest using the code above, this model will be used as an analysis

![image](https://github.com/Chengalex96/MIE1624_Assignment_3/assets/81919159/d13cfa1b-3cf8-4428-bba3-8f267d2181d4)

This model got an accuracy of 63.58% for the liberal party

![image](https://github.com/Chengalex96/MIE1624_Assignment_3/assets/81919159/09e89836-3b6e-420e-9d6a-d36e09ad2c0e)

This model got an accuracy of 63.76% for the conservative party

![image](https://github.com/Chengalex96/MIE1624_Assignment_3/assets/81919159/ef8e3f8f-aeaa-450f-813d-4204da696044)

This model got an accuracy of 70.0% for the NDP

![image](https://github.com/Chengalex96/MIE1624_Assignment_3/assets/81919159/7a09d39a-e190-49b9-8154-8c6a7bef3c66)

This model got an accuracy of 50.0% for PPC

The NLP analytics based on tweets is useful during election campaigns. 

This model, tells us that Liberals and Conservatives have much more negative comments (the left column shows the actual negative results), since these parties are more popular, they're more prone to controversial tweets and have a wide polarity of supporters and opposition. Our model seems to underestimate the number of negative tweets in the election dataset (it thinks the tweets are positive). 

We can compare this to the actual election results in 2021: The Liberal Party won with the Conservative Party trailing just a bit. The tweet sentiment for NDP is pretty low but very positive and many people are voting for them (19% for NDP vs 31.5% Liberal and 31% Conservatives). PPC has a mixed sentiment, where half the tweets are positive and the other half are negative. (7% of voted chose them)

One method to increase the accuracy of the model is to have more features and include more common words. Currently, it takes the top commonly shared words in both CSV files, so it is unable to distinguish between nonfrequent words. 
Another issue is that if multiple parties are included in a tweet, they are most likely being compared with one another, it's difficult to classify which party is receiving a positive bias and the other negative.

**Part 4b: Examining the reason for negative tweets**:

We see that there are 11 unique negative reasons, we encode all of them from 0-10:

df_negative.negative_reason = pd.Categorical(df_negative.negative_reason)

df_negative['code'] = df_negative.negative_reason.cat.codes

Split the data, check for common words, define common words to be the intersection of the training and test set, keep the 500 most common words

Prepare the data for the bag of words on the training data and test data, and standardize the data

I will be training multi-class classification models to predict the reason for negative tweets

Hyperparameter tuning with Lasso

reg_gridsearch = linear_model.Lasso(tol = 0.001, random_state=0) 
parameters = {'alpha':[0.01,0.05, 0.1,0.5,1,3,5,10]}

Best alpha used is: 0.1

Random forest got an accuracy of 52.99% on the testing set

Hyperparameter Tuning - using GridsearchCV on either linear regression or random forest

![image](https://github.com/Chengalex96/MIE1624_Assignment_3/assets/81919159/e09c2b4d-29e4-4d3f-b35c-a26b621464d7)

This model got an accuracy of 50.43% on the negative set - since the accuracy is lower, the model may be overfitting to the training data

**Low accuracy / fail to predict the correct negative reasons may be due to many factors:**

For example, code 8 (segregation) has no results, and code 6(others) has too many results. To improve the accuracy, we can try using more training data for the model to learn more words associated with each code / negative reason.

Another reason is that the model may be overfitting to the training data and irrelevant words, by placing more stop words, the model can emphasize/focus on the more meaningful words. 

Another method is to try using different models and apply hypertuning to them.

Found the Negative top 50 words for negative election sentiment and the positive top 50 words for positive election sentiment

Ranking the top 50 most frequent non-stop words in the Canadian Election dataset, we can observe that:     

- Relating to Part 4.a.a, the model accuracy on distinguishing between negative and positive election tweets may be low because there are many common words between the positive and negative sentiments tweets that are shared among each other. There are many cases, for example, the word 'like' is commonly seen in both, however, for the negative sentiment, there are a few words (ie. do not like) that can cause a semantic change.
- This helps us observe what words are used as a factor in deciding the sentiment of the tweet.

![image](https://github.com/Chengalex96/MIE1624_Assignment_3/assets/81919159/eaf78101-f3bf-4233-88e9-978997e2305d)
