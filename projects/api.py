from flask import Blueprint, jsonify
from util.knowledge_base import get_projects

projects_bp = Blueprint('projects', __name__)

@projects_bp.route('/projects')
def projects():
    return jsonify(get_projects())