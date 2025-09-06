

from flask import Flask, request
from flask_cors import CORS
from about_me.routes import about_me_bp
from projects.api import projects_bp
from experience.api import experience_bp

# Import chatbot routes with error handling
try:
    from chatbot.routes import chatbot_bp
    CHATBOT_ROUTES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Chatbot routes not available: {e}")
    CHATBOT_ROUTES_AVAILABLE = False

app = Flask(__name__)

# Configure CORS for multiple origins including local development and production
allowed_origins = [
    "https://hamza-18.github.io",
    "https://portfolio-backend-vgir.onrender.com",
    "http://localhost:3000",  # for local frontend development
    "http://localhost:5000",  # for local backend testing
    "http://127.0.0.1:5000"   # alternative local address
]

# More permissive CORS configuration for better GitHub Pages compatibility
CORS(app, 
     resources={r"/*": {"origins": "*"}},  # Allow all origins temporarily for debugging
     supports_credentials=False,
     allow_headers=['Content-Type', 'Authorization', 'Accept', 'Origin', 'X-Requested-With'],
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
)

# Add CORS headers manually as backup for all responses
@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    
    # Always add CORS headers for better compatibility
    response.headers['Access-Control-Allow-Origin'] = origin if origin else '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization,Accept,Origin,X-Requested-With'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    
    # Log for debugging
    print(f"Request from origin: {origin}, CORS header added: {response.headers.get('Access-Control-Allow-Origin')}")
    
    return response

app.register_blueprint(about_me_bp)
app.register_blueprint(projects_bp)
app.register_blueprint(experience_bp)

# Register chatbot routes if available
if CHATBOT_ROUTES_AVAILABLE:
    app.register_blueprint(chatbot_bp, url_prefix='/api/chatbot')
    print("✅ Chatbot routes registered at /api/chatbot")
else:
    print("⚠️ Chatbot routes not registered - install dependencies first")

@app.route('/')
def home():
    return 'Hello, Flask!'


if __name__ == '__main__':
    app.run(debug=True)
