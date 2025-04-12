import pandas as pd
from faker import Faker
import random
from sklearn.decomposition import NMF
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

#Example usage of our model with a fake data generated with the Faker Python library 

fake = Faker()

Faker.seed(42)
random.seed(42)

NUM_USERS = 100
NUM_ITEMS = 200
MAX_LIKES_PER_USER = 20
MAX_COMMENTS_PER_USER = 15

users = []
for i in range(1, NUM_USERS + 1):
    users.append({
        'Id': i,
        'Email': fake.unique.email(),
        'Password': fake.password(length=12),
        'Is_verified': fake.boolean(chance_of_getting_true=80),
        'Role': random.choice(['user', 'admin', 'moderator']),
        'Name': fake.name(),
        'Type_of_company': fake.company()
    })

users_df = pd.DataFrame(users)

items = []
categories = ['Electronics', 'Furniture', 'Clothing', 'Books', 'Food', 'Vehicles', 'Real Estate']
statuses = ['Available', 'Sold', 'Reserved', 'Expired']

for i in range(1, NUM_ITEMS + 1):
    expiry_date = fake.date_between(start_date='today', end_date='+2y')
    items.append({
        'Id': i,
        'Status': random.choice(statuses),
        'Category': random.choice(categories),
        'Price': round(random.uniform(1, 10000), 2),
        'Expiry_date': expiry_date,
        'Location': random.randint(10000, 99999),
        'User_id': random.randint(1, NUM_USERS),
        'Title': fake.sentence(nb_words=5),
        'Content': fake.paragraph(nb_sentences=3)
    })

items_df = pd.DataFrame(items)

likes = []
like_id = 1

for user_id in range(1, NUM_USERS + 1):
    num_likes = random.randint(0, MAX_LIKES_PER_USER)
    liked_items = random.sample(range(1, NUM_ITEMS + 1), min(num_likes, NUM_ITEMS))
    
    for item_id in liked_items:
        likes.append({
            'Id': like_id,
            'User_id': user_id,
            'Item_id': item_id
        })
        like_id += 1

likes_df = pd.DataFrame(likes)

comments = []
comment_id = 1

for user_id in range(1, NUM_USERS + 1):
    num_comments = random.randint(0, MAX_COMMENTS_PER_USER)
    commented_items = random.sample(range(1, NUM_ITEMS + 1), min(num_comments, NUM_ITEMS))
    
    for item_id in commented_items:
        comments.append({
            'Id': comment_id,
            'User_id': user_id,
            'Item_id': item_id,
            'Content': fake.paragraph(nb_sentences=2)
        })
        comment_id += 1

comments_df = pd.DataFrame(comments)

# Save to CSV files (optional)
users_df.to_csv('users.csv', index=False)
items_df.to_csv('items.csv', index=False)
likes_df.to_csv('likes.csv', index=False)
comments_df.to_csv('comments.csv', index=False)

print("Data generation complete!")
print(f"Generated {len(users_df)} users, {len(items_df)} items, {len(likes_df)} likes, and {len(comments_df)} comments.")

def create_interaction_matrix(likes_df, comments_df, num_users, num_items):
    likes_matrix = pd.DataFrame(0, index=range(1, num_users+1), columns=range(1, num_items+1))
    comments_matrix = pd.DataFrame(0, index=range(1, num_users+1), columns=range(1, num_items+1))
    
    if not likes_df.empty:
        likes_counts = likes_df.groupby(['User_id', 'Item_id']).size().reset_index(name='Count')
        for _, row in likes_counts.iterrows():
            likes_matrix.at[row['User_id'], row['Item_id']] = row['Count']
    
    if not comments_df.empty:
        comments_counts = comments_df.groupby(['User_id', 'Item_id']).size().reset_index(name='Count')
        for _, row in comments_counts.iterrows():
            comments_matrix.at[row['User_id'], row['Item_id']] = row['Count']
    
    interaction_matrix = likes_matrix + (comments_matrix * 0.5)
    
    return interaction_matrix

def create_content_features(items_df):
    if items_df.empty:
        return np.array([])
    
    items_df['text'] = items_df['Title'].fillna('') + ' ' + items_df['Content'].fillna('')
    tfidf = TfidfVectorizer(stop_words='english', max_features=100)
    try:
        text_features = tfidf.fit_transform(items_df['text'])
    except:
        text_features = np.zeros((len(items_df), 1))
    
    mlb = MultiLabelBinarizer()
    try:
        category_features = mlb.fit_transform(items_df['Category'].apply(lambda x: [x] if pd.notna(x) else ['unknown']))
    except:
        category_features = np.zeros((len(items_df), 1))
    
    return np.hstack([category_features, text_features.toarray()])

def train_model(interaction_matrix, content_features, n_factors=15):
    if interaction_matrix.empty or interaction_matrix.sum().sum() == 0:
        raise ValueError("Interaction matrix is empty")
    
    model = NMF(
        n_components=n_factors,
        init='nndsvd',
        random_state=42
    )
    
    user_factors = model.fit_transform(interaction_matrix)
    item_factors = model.components_.T
    
    return user_factors, item_factors

def recommend_items(user_id, user_factors, item_factors, items_df, likes_df, comments_df, n=5):
    if user_id > user_factors.shape[0] or user_factors.shape[0] == 0:
        return get_popular_items(items_df, likes_df, n=n)
    
    user_vector = user_factors[user_id - 1]
    
    scores = np.dot(item_factors, user_vector)
    
    recommendations = pd.DataFrame({
        'Item_id': range(1, len(scores) + 1),
        'Score': scores
    }).merge(items_df, left_on='Item_id', right_on='Id', how='left')
    
    user_likes = set(likes_df[likes_df['User_id'] == user_id]['Item_id']) if not likes_df.empty else set()
    user_comments = set(comments_df[comments_df['User_id'] == user_id]['Item_id']) if not comments_df.empty else set()
    recommendations = recommendations[~recommendations['Item_id'].isin(user_likes.union(user_comments))]
    
    return recommendations.sort_values('Score', ascending=False).head(n)

def get_popular_items(items_df, likes_df, n=5):
    if likes_df.empty:
        return items_df.sample(min(n, len(items_df)))
    return items_df[items_df['Id'].isin(likes_df['Item_id'].value_counts().head(n).index)].head(n)

def evaluate_model(user_factors, item_factors, test_likes_df, test_comments_df, train_likes_df, train_comments_df, items_df, top_n=5):
    all_precisions = []
    all_recalls = []
    all_accuracies = []

    num_users = user_factors.shape[0]
    all_items = set(items_df['Id'])

    for user_id in range(1, num_users + 1):
        true_likes = set(test_likes_df[test_likes_df['User_id'] == user_id]['Item_id'])
        true_comments = set(test_comments_df[test_comments_df['User_id'] == user_id]['Item_id'])
        true_interacted = true_likes.union(true_comments)

        if not true_interacted:
            continue 

        train_likes = set(train_likes_df[train_likes_df['User_id'] == user_id]['Item_id'])
        train_comments = set(train_comments_df[train_comments_df['User_id'] == user_id]['Item_id'])
        train_interacted = train_likes.union(train_comments)

        recs = recommend_items(user_id, user_factors, item_factors, items_df, train_likes_df, train_comments_df, n=top_n)
        recommended_items = set(recs['Item_id'].values)

        TP = len(recommended_items & true_interacted)
        FP = len(recommended_items - true_interacted)
        FN = len(true_interacted - recommended_items)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        y_true = [1 if i in true_interacted else 0 for i in items_df['Id']]
        y_pred = [1 if i in recommended_items else 0 for i in items_df['Id']]
        accuracy = accuracy_score(y_true, y_pred)

        all_precisions.append(precision)
        all_recalls.append(recall)
        all_accuracies.append(accuracy)

    # Average metrics across users
    avg_precision = np.mean(all_precisions)
    avg_recall = np.mean(all_recalls)
    avg_accuracy = np.mean(all_accuracies)

    print(f"\nEvaluation Results:")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall:    {avg_recall:.4f}")
    print(f"Accuracy:  {avg_accuracy:.4f}")


if __name__ == "__main__":
    NUM_USERS = 100
    NUM_ITEMS = 200

    train_likes_df, test_likes_df = train_test_split(likes_df, test_size=0.2, random_state=42)
    train_comments_df, test_comments_df = train_test_split(comments_df, test_size=0.2, random_state=42)

    interaction_matrix = create_interaction_matrix(train_likes_df, train_comments_df, NUM_USERS, NUM_ITEMS)
    content_features = create_content_features(items_df)

    user_factors, item_factors = train_model(interaction_matrix, content_features)

    test_users = [1, 2, 3, 4, 5]
    for user_id in test_users:
        print(f"\nRecommendations for User {user_id}:")
        try:
            user_name = users_df.loc[users_df['Id'] == user_id, 'Name'].values[0]
            print(f"Name: {user_name}")
        except:
            print("Name: Unknown")
        
        recs = recommend_items(user_id, user_factors, item_factors, items_df, train_likes_df, train_comments_df)
        print(recs[['Item_id', 'Title', 'Category', 'Score']].to_string(index=False) if not recs.empty else "No recommendations")

    # Evaluate precision, recall, accuracy
    evaluate_model(user_factors, item_factors, test_likes_df, test_comments_df,
                   train_likes_df, train_comments_df, items_df)