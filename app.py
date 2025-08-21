

from flask import Flask
from flask_cors import CORS
from about_me.routes import about_me_bp



app = Flask(__name__)
CORS(app)
app.register_blueprint(about_me_bp)

@app.route('/')
def home():
    return 'Hello, Flask!'


if __name__ == '__main__':
    app.run(debug=True)
