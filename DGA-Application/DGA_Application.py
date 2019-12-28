"""
Author: Suyash Sathe
Date: 12 December 2019
Contact: 0470372165
Email: suyash0106@outlook.com

Summary: This is a web application for identification of domain names generated from Domain Generating Algorithms (DGAs). This web 
	 application is developed using Flask Framework which is used to integrate machine learning models into a web application.
"""

# Load the following set of libraries and packages
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

"""
	Function to tokenize the URL
	@param self, url
	@returns tokenised url
"""
def makeTokens(f):

	# make tokens after splitting by slash
	tkns_BySlash = f.split('/')	
	total_Tokens = []
	
	for i in tkns_BySlash:	
		# make tokens after splitting by dash
		tokens = str(i).split('-')	
		tkns_ByDot = []
		
		for j in range(0,len(tokens)):
			# make tokens after splitting by dot
			temp_Tokens = str(tokens[j]).split('.')	
			tkns_ByDot = tkns_ByDot + temp_Tokens
		total_Tokens = total_Tokens + tokens + tkns_ByDot
	
	#remove redundant tokens
	total_Tokens = list(set(total_Tokens))	
	
	#removing .com since it occurs a lot of times and it should not be included in our features
	if 'com' in total_Tokens:
		total_Tokens.remove('com')
		
	return total_Tokens


"""
	Generalized function to create a sparse matrix
	@param self
	@returns CountVectorizer cv
"""
def vectorize():
	urls_data= pd.read_csv("dga_domains.csv")

	# Features
	url_list = urls_data["host"]
	
	# Extract Feature With CountVectorizer
	cv = CountVectorizer(tokenizer=makeTokens)
	
	# Fit the Data
	X = cv.fit_transform(url_list) 
	
	return cv

	
# Give the name of current module (__name__) as the argument
app = Flask(__name__)


"""
	Associates "/" (home) page url with the function and renders 'home.html' page
	@param self
	@returns render_template('home.html')
"""
@app.route('/')
def home():
	return render_template('home.html')

	
"""
	Associates "/" (home) page url with the function and renders 'home.html' page.
	Predict the class of the domain name using logistic regression model model and displays the result on the 'home.html' page.
	@param self
	@returns render_template('home.html', prediction)
"""
@app.route('/',methods=['POST'])
def predict_dga():
	
	# Create a sparse matrix using function vectorize
	cv = vectorize()
	
	# load the logistic regression model from the disk for predicting DGA
	logit_model = pickle.load(open('logit_model.sav', 'rb'))	
	
	# Get the url from the home page
	if request.method == 'POST':
		url = request.form['url']
		len_url = len(makeTokens(url))
		
		# Check for a valid URL. If the URL has less than 2 components, it is an invalid URL.
		if len_url < 2:
			my_prediction = "invalid url"
		else: 
			
			# Transform the URL to its vector form and predict its class
			data = [url]
			vect = cv.transform(data).toarray()
			my_prediction = logit_model.predict(vect)
	return render_template('home.html',prediction = my_prediction)

	
	
"""
	Associates "/subclass" (subclass) page url with the function and renders 'subclass.html' page
	@param self
	@returns render_template('subclass.html')
"""
@app.route('/subclass',methods=['POST'])
def subclass():
	return render_template('subclass.html')
	
	
"""
	Associates "/subclass_result" (subclass) page url with the function and renders 'subclass.html' page.
	Predict the subclass of the domain name using Multiclass Logistic Regression model and displays the result on the 'subclass.html' page.
	@param self
	@returns render_template('subclass.html', prediction)
"""
@app.route('/subclass_result',methods=['POST'])
def predict_subclass():
	
	cv = vectorize()
	
	# load the multiclass logistic regression model from disk for predicting the subclass of the domain name
	multiclass_logit_model = pickle.load(open('multiclass_logit_model.sav', 'rb'))	
	
	# Get the url from the home page	
	if request.method == 'POST':
		url = request.form['url']
		len_url = len(makeTokens(url))
		
		# Check for a valid URL. If the URL has less than 2 components, it is an invalid URL.
		if len_url < 2:
			my_prediction = "invalid url"
		else: 
			# Transform the URL to its vector form and predict its class
			data = [url]
			vect = cv.transform(data).toarray()
			my_prediction = multiclass_logit_model.predict(vect)
	return render_template('subclass.html',subclass_prediction = my_prediction)


# Calling the main method and run the application	
if __name__ == '__main__':
	app.run(debug=True)
