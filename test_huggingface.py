#!/usr/bin/env python3
"""
Test HuggingFace API integration with the chatbot
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_huggingface_chatbot():
    print("🤗 TESTING HUGGINGFACE-POWERED CHATBOT")
    print("=" * 60)
    
    try:
        from chatbot.chatbot_engine import RAGChatbot
        
        print("Initializing chatbot with HuggingFace API...")
        chatbot = RAGChatbot()
        print("✅ Chatbot initialized successfully!")
        
        # Test queries
        test_queries = [
            "Hello!",
            "Tell me about Hamza's work experience",
            "What projects has he worked on?",
            "Tell me about his time at Wavelet",
            "What is his educational background?"
        ]
        
        print("\nTesting queries:")
        print("-" * 40)
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            try:
                response = chatbot.chat(query)
                print(f"Response: {response[:150]}...")
                print("✅ Query processed successfully")
            except Exception as e:
                print(f"❌ Query failed: {e}")
                
    except Exception as e:
        print(f"❌ Chatbot initialization failed: {e}")
        return False
    
    print("\n🎉 HuggingFace integration test completed!")
    return True

if __name__ == "__main__":
    test_huggingface_chatbot()
