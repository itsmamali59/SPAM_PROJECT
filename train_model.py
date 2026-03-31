print("START")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
texts = [
    'Win money now',
    'Hello friend how are you',
    'Free offer just for you',
    'Meeting at 5pm',
    'Claim your prize now',
    'Let’s study together']
labels = [1, 0, 1, 0, 1, 0]
cv = CountVectorizer()
X = cv.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(cv, open('vectorizer.pkl', 'wb'))
print("Model Ready!")











