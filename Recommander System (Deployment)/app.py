from flask import Flask, jsonify, request
from .model import get_recommendations_for_user, train_recommendation_model
from .cache import load_cached_model
from .config import RECOMMENDATION_SETTINGS

app = Flask(__name__)

load_cached_model()

@app.route('/api/recommendations', methods=['GET'])
def api_get_recommendations():
    """API endpoint to get recommendations for a user"""
    try:
        user_id = int(request.args.get('user_id', 0))
        if user_id <= 0:
            raise ValueError("Invalid user ID")
        
        n = int(request.args.get('n', RECOMMENDATION_SETTINGS['default_recommendations']))
        n = min(n, RECOMMENDATION_SETTINGS['max_recommendations'])
        
        recommendations = get_recommendations_for_user(user_id, n)
        
        response = {
            'success': True,
            'recommendations': [{
                'id': item['id'],
                'title': item['title'],
                'description': item.get('description', ''),
                'category': item.get('category', ''),
                'image_url': item.get('image_url', '')
            } for item in recommendations]
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/refresh_model', methods=['POST'])
def api_refresh_model():
    """API endpoint to manually refresh the model"""
    try:
        result = train_recommendation_model()
        if result:
            return jsonify({'success': True, 'message': 'Model refreshed successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to refresh model'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)