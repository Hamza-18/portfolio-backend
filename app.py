

from flask import Flask
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
CORS(app, resources={r"/api/*": {"origins": "https://hamza-18.github.io"}})
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
