import json
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load intents
with open("data/intents.json") as f:
    intents = json.load(f)

sentences = []
labels = []

for intent, phrases in intents.items():
    for phrase in phrases:
        sentences.append(phrase)
        labels.append(intent)

# Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)
y = labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("NLU Model Accuracy:", accuracy_score(y_test, y_pred))

responses = {
    "greeting": [
        "Hello! Ready for some cricket talk? 🏏",
        "Hi! Let's discuss today's match."
    ],
    "match_query": [
        "India is playing against Australia today.",
        "Today's match is a thrilling one!"
    ],
    "score_query": [
        "India is 145/3 in 20 overs.",
        "The current score is 210/5."
    ],
    "player_query": [
        "Virat Kohli is batting brilliantly.",
        "Rohit Sharma is in great form today."
    ],
    "emotion_happy": [
        "Absolutely! That was a fantastic shot! 🔥",
        "Crowd is going wild!"
    ],
    "emotion_sad": [
        "Yeah, that was unfortunate.",
        "Tough moment for the team."
    ],
    "goodbye": [
        "Bye! Enjoy the match! 🏏",
        "See you later!"
    ]
}

print("🏏 Cricket Conversation Chatbot Started (type 'exit' to stop)")

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        print("Bot: Match over! Goodbye 👋")
        break

    vector = vectorizer.transform([user_input])
    intent = model.predict(vector)[0]
    print("Bot:", random.choice(responses[intent]))

