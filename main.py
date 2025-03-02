import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

temp = pd.read_csv('spam_ham_dataset.csv')
data = pd.DataFrame(temp)

x = data['text']
y = data['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

model = LogisticRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy of model: {accuracy:.2f}")
print("Звіт класифікації:\n", classification_report(y_test, y_pred))

probabilities = model.predict_proba(x_test)
plt.figure(figsize=(8, 4))
plt.bar(range(len(y_test)), probabilities[:, 1], color='blue', alpha=0.6, label="Ймовірність класу 1")
plt.xlabel("Зразки")
plt.ylabel("Ймовірність")
plt.legend()
plt.show()


