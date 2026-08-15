import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# --- 1. MODEL TRAINING ---
@st.cache_resource
def train_spam_model():
    # Kaggle se download ki hui file ka path
    # Agar encoding error aaye toh encoding='latin-1' lagaya hai
    df = pd.read_csv("spam.csv", encoding="latin-1")

    # Kaggle ke dataset ke hisab se columns clean karna
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]

    # ham (normal message) ko 0 aur spam ko 1 mein convert karna
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

    X = df["message"]
    y = df["label_num"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Text ko numbers mein convert karne ke liye TF-IDF Vectorizer
    cv = TfidfVectorizer()
    X_train_vec = cv.fit_transform(X_train)

    # Train Machine Learning Model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    return model, cv


# Model load karna
model, cv = train_spam_model()

# --- 2. WEB APP UI (STREAMLIT) ---
st.title("🛡️ AI Spam Message Detector")
st.write("Apna message niche paste karein aur check karein ki ye Spam hai ya Safe.")

# User input text box
user_input = st.text_area("Enter Message Here:", height=120)

if st.button("Check Message"):
    if user_input.strip() != "":
        data = [user_input]
        vect = cv.transform(data).toarray()
        prediction = model.predict(vect)

        if prediction == 1:
            st.error("🚨 Warning: Yeh ek SPAM message hai!")
        else:
            st.success("✅ Safe: Yeh ek normal (LEGITIMATE) message hai.")
    else:
        st.warning("Kripya pehle box mein koi text type karein.")