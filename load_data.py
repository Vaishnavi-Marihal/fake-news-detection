import pandas as pd

# Load datasets
true_df = pd.read_csv('data/True.csv')
fake_df = pd.read_csv('data/Fake.csv')

# Add label: 1 = Real, 0 = Fake
true_df['label'] = 1
fake_df['label'] = 0

# Keep only title + text + label
true_df = true_df[['title', 'text', 'label']]
fake_df = fake_df[['title', 'text', 'label']]

# Combine both datasets
df = pd.concat([true_df, fake_df], ignore_index=True)

# Combine title and text into one column
df['combined_text'] = df['title'] + " " + df['text']

# Remove rows with missing text
df = df.dropna()

# Save cleaned dataset (optional)
df.to_csv('data/cleaned_news.csv', index=False)

print("✅ Data loaded successfully!")
print(f"Total articles: {len(df)}")
print(f"Real news: {sum(df['label'] == 1)}")
print(f"Fake news: {sum(df['label'] == 0)}")