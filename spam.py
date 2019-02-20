import pandas as pd
def removeStopwords(x):
	from nltk.tokenize import RegexpTokenizer
	from nltk.corpus import stopwords
	stopWords=set(stopwords.words('english'))
	tokenizer=RegexpTokenizer(r'\w+')
	tokens=[tokenizer.tokenize(i) for i in x]
	filtered_tokens=list()
	for token in tokens:
		filtered_tokens.append([word for word in token if word not in stopWords])
	return filtered_tokens
def encodeData(tokens):
	tokenized_x=pd.DataFrame(tokens)
	tokenized_x=tokenized_x.fillna("None")
	from sklearn.preprocessing import OneHotEncoder
	enc=OneHotEncoder(handle_unknown='ignore')
	labeled_x=enc.fit_transform(tokenized_x)
	labeled_x=labeled_x.toarray()
	labeled_x=pd.DataFrame(labeled_x)
	return labeled_x
#read the csv file	
data=pd.read_csv("sms_spam.csv")
# the output
y=data['type']
#input
x=data['text']
#print(x.shape)
#print(y.shape)
#tokenize and remove stopwords
tokens=removeStopwords(x)
#encode the categorical data 
labeled_x=encodeData(tokens)

#split the data to train and test sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(labeled_x,y,test_size=0.3,random_state=45)
from sklearn.linear_model import LogisticRegression
log=LogisticRegression(solver='liblinear')
log.fit(x_train,y_train)
print("Accuracy on test data is ",100*log.score(x_test,y_test))
