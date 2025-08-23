from flask import Blueprint, jsonify
from util.knowledge_base import get_experience

experience_bp = Blueprint('experience', __name__)

@experience_bp.route('/experience')
def experience():
    return jsonify(get_experience())