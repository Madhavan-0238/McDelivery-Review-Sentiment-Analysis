import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

# ================= SETUP =================
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()

print("McDonald's Review Analysis\n")

# ================= LOAD DATA =================
df = pd.read_excel("sample_reviews.xlsx")

print("Dataset Shape:", df.shape)
print("Columns:", list(df.columns))
print("\nMissing Values:\n", df.isnull().sum())

# ================= BASIC CLEANING =================
df = df.drop(columns=["Unnamed: 8"], errors="ignore")
df = df.dropna(subset=["order_id", "business_date", "StoreCode", "rider_id"])

print("\nAfter basic cleaning:", df.shape)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

df["clean_review"] = df["remark"].fillna("").apply(clean_text)

print("\nPHASE 2: TEXT PREPROCESSING COMPLETED")
print(df["clean_review"].head())


def get_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["clean_review"].apply(get_sentiment)

print("\nPHASE 3: SENTIMENT ANALYSIS")
print(df["sentiment"].value_counts())

themes = {
    "Delivery Speed": ["late", "delay", "slow", "fast"],
    "Food Quality": ["cold", "fresh", "stale", "quality"],
    "Taste & Flavor": ["tasty", "delicious", "yummy", "bland", "flavor"],
    "Packaging": ["spill", "leak", "package", "damaged", "sealed"],
    "App Experience": ["app", "crash", "login", "bug"],
    "Payments": ["payment", "refund", "charged", "failed"],
    "Customer Support": ["support", "call", "helpline"],

    "Portion Size": ["portion", "quantity", "small", "less", "enough"],
    "Order Accuracy": ["wrong", "missing", "incorrect", "item", "order"],
    "Pricing & Value": ["price", "cost", "expensive", "cheap"],
    "Value Satisfaction": ["worth", "value for money", "waste"],
    "Hygiene & Safety": ["hygiene", "clean", "dirty", "unsafe"],

    "Waiting Time": ["wait", "waiting", "long", "queue"],
    "Service Speed": ["service slow", "counter", "billing"],
    "Overall Experience": ["experience", "overall", "worst", "best"],
    "Emotional Feedback": ["good", "bad", "nice", "okay"],

    "Fries": ["fries", "french fries", "chips"],
    "Burgers": ["burger", "mcchicken", "maharaja", "veg burger"],
    "Beverages": ["coke", "cola", "drink", "cold drink", "beverage"],
   "General Feedback": ["average", "satisfied", "unsatisfied", "okayish"]

}


def detect_theme(text):
    for theme, keywords in themes.items():
        if any(word in text for word in keywords):
            return theme
    return "Other"

df["theme"] = df["clean_review"].apply(detect_theme)

print("\nPHASE 4: THEME EXTRACTION")
print(df["theme"].value_counts())


from collections import Counter

neg = df[df["sentiment"] == "Negative"]
fries_words = ["fries", "french fry", "french fries", "chips"]

fries_neg = neg[neg["clean_review"].str.contains("|".join(fries_words), na=False)]

print("\nPHASE 5: FRIES-RELATED NEGATIVE REVIEWS")
print("Total negative reviews:", len(neg))
print("Negative reviews about fries:", len(fries_neg))
print(f"Percentage: {(len(fries_neg)/len(neg))*100:.2f}%" if len(neg) else "Percentage: 0%")

top_words = Counter(" ".join(fries_neg["clean_review"]).split()).most_common(10)
print("\nTop complaint words related to fries:")
print(top_words)

words, counts = zip(*top_words)

plt.figure()
plt.bar(words, counts)
plt.title("Top Issues in Fries-Related Negative Reviews")
plt.xlabel("Complaint Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ================= THEME vs SENTIMENT SUMMARY =================
print("\nTHEME vs SENTIMENT SUMMARY")
print(pd.crosstab(df["theme"], df["sentiment"]))


sentiment_counts = df["sentiment"].value_counts()

plt.figure()
sentiment_counts.plot(kind="bar")
plt.title("Sentiment Distribution of Reviews")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.show()

# ---- Word Cloud: Positive Reviews ----
positive_text = " ".join(df[df["sentiment"] == "Positive"]["clean_review"])

positive_wc = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(positive_text)

plt.figure()
plt.imshow(positive_wc)
plt.axis("off")
plt.title("Word Cloud - Positive Reviews")
plt.show()

# ---- Word Cloud: Negative Reviews ----
negative_text = " ".join(df[df["sentiment"] == "Negative"]["clean_review"])

negative_wc = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(negative_text)

plt.figure()
plt.imshow(negative_wc)
plt.axis("off")
plt.title("Word Cloud - Negative Reviews")
plt.show()

print("\n===== PHASE 2, 3, 4 & VISUALIZATION COMPLETED SUCCESSFULLY =====")

