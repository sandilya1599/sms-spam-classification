import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
def removeStopwords(x):
	stopWords=set(stopwords.words('english'))
	tokenizer=RegexpTokenizer(r'\w+')
	tokens=[tokenizer.tokenize(i) for i in x]
	filtered_tokens=list()
	for token in tokens:
		filtered_tokens.append([word for word in token if word not in stopWords])
	lemmatizer=WordNetLemmatizer()
	for i in range(len(filtered_tokens)):
		filtered_tokens[i]=[lemmatizer.lemmatize(token) for token in filtered_tokens[i]]	
	return filtered_tokens
def encodeData(tokens):
	tokenized_x=pd.DataFrame(tokens)
	tokenized_x=tokenized_x.fillna("None")
	enc=OneHotEncoder(handle_unknown='ignore')
	labeled_x=enc.fit_transform(tokenized_x)
	labeled_x=labeled_x.toarray()
	labeled_x=pd.DataFrame(labeled_x)
	return labeled_x
def main():
	#read the csv file	
	data=pd.read_csv("sms_spam.csv")
	# the output
	y=data['type']
	#input
	x=data['text']
	#tokenize and remove stopwords
	tokens=removeStopwords(x)
	#encode the categorical data 
	labeled_x=encodeData(tokens)
	#split the data to train and test sets
	x_train,x_test,y_train,y_test=train_test_split(labeled_x,y,test_size=0.5,random_state=45)
	gnb=GaussianNB()
	gnb.fit(x_train,y_train)
	print("The accuracy is {:.4f}".format(gnb.score(x_test,y_test)*100))

if __name__=="__main__":
	main()