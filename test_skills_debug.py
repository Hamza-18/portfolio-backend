#!/usr/bin/env python3
"""
Debug test for skills query
"""

import sys
import os
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot.chatbot_engine import RAGChatbot

def test_skills_query():
    """Test the specific skills query that user says is not working"""
    logging.basicConfig(level=logging.INFO)
    
    print("🔍 DEBUGGING SKILLS QUERY ISSUE")
    print("="*50)
    
    # Initialize chatbot
    try:
        chatbot = RAGChatbot()
        print("✅ Chatbot initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        return
    
    # Test the exact query mentioned by user
    query = "tell me about his skills"
    print(f"\nTesting query: '{query}'")
    print("-" * 30)
    
    try:
        response = chatbot.chat(query)
        print(f"Response length: {len(response)} characters")
        print(f"Response preview: {response[:150]}...")
        
        # Check if response contains expected skill-related content
        skill_indicators = [
            "Python", "Java", "JavaScript", "C++", "Flask", 
            "Node.js", "Angular", "AWS", "Computer Science",
            "software engineering", "machine learning", "programming"
        ]
        
        found_skills = [skill for skill in skill_indicators if skill.lower() in response.lower()]
        
        print(f"\nSkills mentioned in response: {found_skills}")
        
        if found_skills:
            print("✅ Skills query appears to be working correctly")
        else:
            print("❌ Skills query may not be working - no skill-related content found")
            
        print(f"\nFull response:\n{response}")
        
    except Exception as e:
        print(f"❌ Error during query processing: {e}")

if __name__ == "__main__":
    test_skills_query()
