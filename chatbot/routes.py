"""
Flask API routes for RAG Chatbot
"""

from flask import Blueprint, request, jsonify
import logging
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import will work once packages are installed
try:
    from chatbot.chatbot_engine import RAGChatbot
    CHATBOT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Chatbot engine not available: {e}")
    CHATBOT_AVAILABLE = False

chatbot_bp = Blueprint('chatbot', __name__)

# Global chatbot instance
chatbot_instance = None

def initialize_chatbot():
    """Initialize the chatbot instance"""
    global chatbot_instance
    if CHATBOT_AVAILABLE and chatbot_instance is None:
        try:
            chatbot_instance = RAGChatbot()
            logging.info("RAG Chatbot initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize chatbot: {e}")
            chatbot_instance = None

@chatbot_bp.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint for the RAG chatbot"""
    try:
        if not CHATBOT_AVAILABLE:
            return jsonify({
                'error': 'Chatbot dependencies not installed. Please install: pip install -r requirements_rag.txt',
                'status': 'error'
            }), 500
        
        # Initialize chatbot if not already done
        if chatbot_instance is None:
            initialize_chatbot()
            
        if chatbot_instance is None:
            return jsonify({
                'error': 'Chatbot initialization failed',
                'status': 'error'
            }), 500
        
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'error': 'Message is required',
                'status': 'error'
            }), 400
        
        user_message = data['message'].strip()
        if not user_message:
            return jsonify({
                'error': 'Message cannot be empty',
                'status': 'error'
            }), 400
        
        # Get chatbot response
        response = chatbot_instance.chat(user_message)
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
        
    except Exception as e:
        logging.error(f"Chat endpoint error: {e}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@chatbot_bp.route('/status', methods=['GET'])
def status():
    """Get chatbot status"""
    return jsonify({
        'chatbot_available': CHATBOT_AVAILABLE,
        'initialized': chatbot_instance is not None,
        'status': 'ready' if (CHATBOT_AVAILABLE and chatbot_instance) else 'not_ready'
    })

@chatbot_bp.route('/initialize', methods=['POST'])
def initialize():
    """Manually initialize the chatbot"""
    try:
        if not CHATBOT_AVAILABLE:
            return jsonify({
                'error': 'Chatbot dependencies not installed',
                'status': 'error'
            }), 500
        
        force_rebuild = request.get_json().get('force_rebuild', False) if request.get_json() else False
        
        global chatbot_instance
        chatbot_instance = RAGChatbot()
        chatbot_instance.initialize(force_rebuild=force_rebuild)
        
        return jsonify({
            'message': 'Chatbot initialized successfully',
            'status': 'success'
        })
        
    except Exception as e:
        logging.error(f"Initialization error: {e}")
        return jsonify({
            'error': f'Initialization failed: {str(e)}',
            'status': 'error'
        }), 500

@chatbot_bp.route('/test', methods=['GET'])
def test():
    """Test endpoint to verify chatbot functionality"""
    try:
        # Test the actual RAGChatbot that's being used
        test_chatbot = RAGChatbot()
        
        # Count the loaded documents from vector store
        vector_store = getattr(test_chatbot, 'vector_store', None)
        if vector_store and hasattr(vector_store, 'embeddings'):
            total_docs = len(vector_store.embeddings)
        else:
            total_docs = 0
        
        # Test a simple query
        test_response = test_chatbot.chat("Hello")
        
        return jsonify({
            'message': 'Chatbot test successful',
            'total_documents': total_docs,
            'test_query': 'Hello',
            'test_response': test_response[:100] + '...' if len(test_response) > 100 else test_response,
            'status': 'success'
        })
        
    except Exception as e:
        logging.error(f"Test endpoint error: {e}")
        return jsonify({
            'error': f'Test failed: {str(e)}',
            'status': 'error'
        }), 500
