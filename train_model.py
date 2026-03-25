import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load training data
X_train, X_test, y_train, y_test = joblib.load('models/split_data.pkl')

# Define models to try
models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(kernel='linear'),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

best_model = None
best_acc = 0

print("⏳ Training models...\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"{name}: {acc:.4f}")
    
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

print(f"\n🏆 Best Model: {best_name} with Accuracy: {best_acc:.4f}")

# Save the best model
joblib.dump(best_model, 'models/fake_news_model.pkl')
print("✅ Best model saved as 'fake_news_model.pkl'")