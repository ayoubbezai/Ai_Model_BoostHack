import mysql.connector
from mysql.connector import Error
from .config import DB_CONFIG
import time

_db_connection_pool = None

def initialize_db_pool():
    global _db_connection_pool
    if _db_connection_pool is None:
        try:
            _db_connection_pool = mysql.connector.pooling.MySQLConnectionPool(
                pool_name="recommender_pool",
                pool_size=5,
                **DB_CONFIG
            )
        except Error as e:
            print(f"Error creating connection pool: {e}")
            raise

def get_db_connection():
    global _db_connection_pool
    if _db_connection_pool is None:
        initialize_db_pool()
    
    attempts = 0
    max_attempts = 3
    wait_time = 1
    
    while attempts < max_attempts:
        try:
            return _db_connection_pool.get_connection()
        except Error as e:
            attempts += 1
            if attempts == max_attempts:
                raise
            time.sleep(wait_time)
            wait_time *= 2

def fetch_recommendation_data():
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT id, title, description, category, image_url, created_at 
            FROM items 
            WHERE status = 'available'
            ORDER BY created_at DESC
        """)
        items = cursor.fetchall()
        
        cursor.execute("SELECT user_id, item_id FROM likes")
        likes = cursor.fetchall()
        
        cursor.execute("SELECT user_id, item_id FROM comments")
        comments = cursor.fetchall()
        
        cursor.execute("SELECT COUNT(*) as count FROM users")
        user_count = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM items WHERE status = 'available'")
        item_count = cursor.fetchone()['count']
        
        return {
            'items': items,
            'likes': likes,
            'comments': comments,
            'user_count': user_count,
            'item_count': item_count,
            'timestamp': time.time()
        }
        
    except Error as e:
        print(f"Database error: {e}")
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()