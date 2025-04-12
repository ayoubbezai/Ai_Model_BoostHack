import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

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

