import os
import pickle
import time
from datetime import datetime, timedelta
from .config import RECOMMENDATION_SETTINGS

_cached_model = None
_last_refresh_time = None

def get_cache_path():
    cache_dir = RECOMMENDATION_SETTINGS['cache_dir']
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return os.path.join(cache_dir, 'recommender_model.pkl')

def load_cached_model():
    global _cached_model, _last_refresh_time
    
    cache_file = get_cache_path()
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
            _cached_model = data['model']
            _last_refresh_time = data['timestamp']
            print("Loaded model from cache")
            return _cached_model
    except Exception as e:
        print(f"Error loading cached model: {e}")
        return None

def save_model_to_cache(model_data):
    global _cached_model, _last_refresh_time
    
    cache_file = get_cache_path()
    _cached_model = model_data
    _last_refresh_time = time.time()
    
    data = {
        'model': _cached_model,
        'timestamp': _last_refresh_time
    }
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Error saving model to cache: {e}")

def model_needs_refresh():
    global _last_refresh_time
    
    if _last_refresh_time is None:
        return True
    
    refresh_interval = timedelta(hours=RECOMMENDATION_SETTINGS['model_refresh_hours'])
    return datetime.now() - datetime.fromtimestamp(_last_refresh_time) > refresh_interval

def get_cached_model():
    global _cached_model
    return _cached_model