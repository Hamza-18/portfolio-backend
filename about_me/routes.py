from flask import Blueprint, jsonify
from util.knowledge_base import get_about_me

about_me_bp = Blueprint('about_me', __name__)

@about_me_bp.route('/about-me')
def about_me():
    return jsonify(get_about_me())