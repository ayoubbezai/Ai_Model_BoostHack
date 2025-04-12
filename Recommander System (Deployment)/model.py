import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from .database import fetch_recommendation_data
from .cache import load_cached_model, save_model_to_cache, model_needs_refresh, get_cached_model
from .config import RECOMMENDATION_SETTINGS
import time

def create_interaction_matrix(likes, comments, num_users, num_items):
    """Create user-item interaction matrix"""
    likes_matrix = np.zeros((num_users, num_items))
    comments_matrix = np.zeros((num_users, num_items))
    
    for like in likes:
        user_idx = like['user_id'] - 1  
        item_idx = like['item_id'] - 1  
        likes_matrix[user_idx][item_idx] += 1
    
    for comment in comments:
        user_idx = comment['user_id'] - 1
        item_idx = comment['item_id'] - 1
        comments_matrix[user_idx][item_idx] += 1
    
    # Combine interactions (likes + 0.5*comments)
    return likes_matrix + (comments_matrix * 0.5)

def create_content_features(items):
    """Create content-based features for items"""
    if not items:
        return np.array([])
    
    texts = [f"{item['title'] or ''} {item['description'] or ''}".strip() for item in items]
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=100)
    try:
        text_features = tfidf.fit_transform(texts).toarray()
    except:
        text_features = np.zeros((len(items), 1))
    
    mlb = MultiLabelBinarizer()
    try:
        categories = [[item['category']] if item['category'] else ['unknown'] for item in items]
        category_features = mlb.fit_transform(categories)
    except:
        category_features = np.zeros((len(items), 1))
    
    return np.hstack([category_features, text_features])

def train_recommendation_model():
    """Train the recommendation model"""
    print("Training recommendation model...")
    start_time = time.time()
    
    data = fetch_recommendation_data()
    if data is None:
        print("Failed to fetch data for training")
        return None
    
    interaction_matrix = create_interaction_matrix(
        data['likes'], 
        data['comments'], 
        data['user_count'], 
        data['item_count']
    )
    
    content_features = create_content_features(data['items'])
    
    try:
        if interaction_matrix.size == 0 or np.sum(interaction_matrix) == 0:
            raise ValueError("Interaction matrix is empty")
        
        model = NMF(
            n_components=RECOMMENDATION_SETTINGS['num_factors'],
            init='nndsvd',
            random_state=42
        )
        
        user_factors = model.fit_transform(interaction_matrix)
        item_factors = model.components_.T
        
        if content_features.size > 0:
            item_factors = 0.7 * item_factors + 0.3 * content_features
        
        model_data = {
            'user_factors': user_factors,
            'item_factors': item_factors,
            'items': data['items'],
            'timestamp': data['timestamp']
        }
        
        save_model_to_cache(model_data)
        
        print(f"Model trained successfully in {time.time() - start_time:.2f} seconds")
        return model_data
        
    except Exception as e:
        print(f"Model training failed: {e}")
        return None

def get_user_interactions(user_id, likes, comments):
    user_likes = {like['item_id'] for like in likes if like['user_id'] == user_id}
    user_comments = {comment['item_id'] for comment in comments if comment['user_id'] == user_id}
    return user_likes.union(user_comments)

def get_popular_items(n=5):
    """Get most popular items based on likes"""
    data = fetch_recommendation_data()
    if data is None or not data['items']:
        return []
    
    if not data['likes']:
        return data['items'][:n] if len(data['items']) >= n else data['items']
    
    # Count likes per item
    item_likes = {}
    for like in data['likes']:
        item_id = like['item_id']
        item_likes[item_id] = item_likes.get(item_id, 0) + 1
    
    # Sort by like count
    sorted_items = sorted(item_likes.items(), key=lambda x: x[1], reverse=True)
    top_item_ids = [item[0] for item in sorted_items[:n]]
    
    # Return item details
    items_dict = {item['id']: item for item in data['items']}
    return [items_dict[item_id] for item_id in top_item_ids if item_id in items_dict]

def get_recommendations_for_user(user_id, n=5):
    """Get recommendations for a specific user"""
    # Check if we need to refresh the model
    if model_needs_refresh():
        print("Model needs refresh, training new model...")
        train_recommendation_model()
    
    model = get_cached_model()
    if model is None:
        model = train_recommendation_model()
        if model is None:
            return get_popular_items(n)
    
    # Handle new users (user_id beyond our matrix)
    if user_id > len(model['user_factors']) or len(model['user_factors']) == 0:
        return get_popular_items(n)
    
    user_vector = model['user_factors'][user_id - 1]
    
    scores = np.dot(model['item_factors'], user_vector)
    
    recommendations = []
    for idx, score in enumerate(scores):
        item_id = idx + 1
        item = next((i for i in model['items'] if i['id'] == item_id), None)
        if item:
            recommendations.append({
                'item': item,
                'score': score
            })
    
    data = fetch_recommendation_data()
    if data is None:
        interacted_items = set()
    else:
        interacted_items = get_user_interactions(user_id, data['likes'], data['comments'])
    
    recommendations = [r for r in recommendations if r['item']['id'] not in interacted_items]
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    
    return [r['item'] for r in recommendations[:n]]