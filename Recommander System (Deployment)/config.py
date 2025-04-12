DB_CONFIG = {
    'host': 'localhost',
    'user': 'your_db_username',
    'password': 'your_db_password',
    'database': 'your_database_name'
}

RECOMMENDATION_SETTINGS = {
    'num_factors': 15,               
    'default_recommendations': 5,     
    'max_recommendations': 20,        
    'model_refresh_hours': 24,        
    'cache_dir': 'recommender_cache'  
}
