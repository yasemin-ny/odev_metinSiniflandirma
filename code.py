import pandas as pd
from sklearn.metrics import accuracy_score
import nltk
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from textblob import Word
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
# Attention: This code contains parts that may not function correctly.
# Any missing functionality or errors will be addressed and fixed in upcoming updates.
# Please bear with me as I work on improving it.
nltk.download('stopwords')
csv_bbc_news=pd.read_csv("bbc-news-data.csv", sep="\t")
csv_bbc_news.head()
csv_bbc_news['category'].unique()
csv_bbc_news['category'].value_counts()
csv_bbc_news_new=pd.DataFrame(csv_bbc_news,columns=['category','content'])
csv_bbc_news_new['content']=csv_bbc_news_new['content'].apply(lambda x: ' '.join(x.lower() for x in x.split()))
csv_bbc_news_new['content']=csv_bbc_news_new['content'].str.replace('[^\w\s]','')
csv_bbc_news_new['content']=csv_bbc_news_new['content'].str.replace('\d','')
sw = stopwords.words('english')
csv_bbc_news_new['content']=csv_bbc_news_new['content'].apply(lambda x: ' '.join(x for x in x.split() if x not in sw))
nltk.download('wordnet')
nltk.download('omw-1.4')
csv_bbc_news_new['content']=csv_bbc_news_new['content'].apply(lambda x: ' '.join([Word(i).lemmatize() for i in x.split()]))
csv_bbc_news_new=csv_bbc_news_new.dropna()
X_train, X_test, y_train, y_test = train_test_split(csv_bbc_news_new['content'], csv_bbc_news_new['category'], test_size = 0.2,random_state=42)
rfc =  Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('rfc', RandomForestClassifier(n_estimators=500,)),])
?RandomForestClassifier
rfc
rfc_model=rfc.fit(X_train,y_train)
y_pred_rfc=rfc_model.predict(X_test)
accuracy_score(y_test,y_pred_rfc)
y_test[:10]
print(classification_report(y_test,y_pred_rfc))
y_pred_rfc[:10]
#clf = Pipeline([RandomForestClassifier(n_estimators = 500, max_depth = 4,warm_start=False, max_features = 3, bootstrap = True, random_state = 18)]).fit(X_train, y_train)

def preprocess(news_text):
    news_text=pd.Series([news_text])
    news_text=news_text.apply(lambda x: ' '.join(x.lower() for x in x.split()))
    news_text=news_text.replace('[^\w\s]','')
    news_text=news_text.replace('\d','')
    sw = stopwords.words('english')
    news_text=news_text.apply(lambda x: ' '.join(x for x in x.split() if x not in sw))
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    news_text=news_text.apply(lambda x: ' '.join([Word(i).lemmatize() for i in x.split()]))
    return news_text

asdsd=preprocess("Canada dominated large portions of this Group F game but were wasteful in front of goal, particularly when Alphonso Davies saw a first-half penalty saved by Belgium keeper Thibaut Courtois.Courtois also had to save well from Alistair Johnston, but Canada were undone against the run of play on the stroke of half-time when Michy Batshuayi collected Toby Alderweireld's long ball and fired a powerful left-foot finish past Milan Borjan.Jonathan David wasted a glorious headed chance to draw Canada level and Courtois also saved from Cyle Larin. Canada were also left nursing a sense of injustice after they had two presentable penalty appeals ignored in the first half.Canada continued to push forward in the second half but it was Roberto Martinez's side who closed out the win, despite a performance that made a mockery of their status as second in the world rankings.Canada deserved to be better than us in the way they played, Martinez told Match of the Day. Its a win and we need to play better and to grow.This tournament is going to make you develop and grow as the tournament goes on. If you do that by winning games, its an incredible advantage.Today we didnt win by our normal talent and quality on the ball, but you dont win in the World Cup if you dont do the other side of the game.")
asdsd[0]
#
test_new_bbc_a=preprocess(test_new_bbc_a)
rfc_model.predict([test_new_bbc_a[0]])
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
!pip install --upgrade scikit-learn
ConfusionMatrixDisplay.from_estimator(rfc_model,X_test,y_test)
plt.show()
test1="Cryptocurrency firms Gemini and Genesis have been charged by US regulators with illegally selling crypto assets to hundreds of thousands of investors.The companies are accused of breaking the law by offering and selling the products through their joint programme, Gemini Earn, which launched in 2021.The Securities and Exchange Commission (SEC) is in charge of the case.Gemini was co-founded by twins Tyler and Cameron Winklevoss - known for their legal dispute with Facebook.Tyler called the complaint 'disappointing', and said his company looks forward to defending itself.Genesis, which is owned by the crypto conglomerate Digital Currency Group, has so far not commented on the charges."
test1=preprocess(test1)
rfc_model.predict([test1[0]])



