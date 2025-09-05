#!/usr/bin/env python3
"""
Test HuggingFace API integration with the chatbot
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot.chatbot_engine import RAGChatbot

def test_huggingface_chatbot():
    print("🤗 TESTING HUGGINGFACE-POWERED CHATBOT")
    print("=" * 60)
    
    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    # Verify API token is available
    if not os.environ.get('HUGGINGFACE_API_TOKEN'):
        print("❌ HUGGINGFACE_API_TOKEN not found in environment variables")
        print("   Please make sure .env file exists with: HUGGINGFACE_API_TOKEN=your_token_here")
        return False
    
    try:
        print("Initializing chatbot with HuggingFace API...")
        chatbot = RAGChatbot()
        print("✅ Chatbot initialized successfully!")
        
        # Test queries
        test_queries = [
            "Hello!",
            "Tell me about Hamza's work experience",
            "What projects has he worked on?", 
            "Tell me about his time at Wavelet",
            "What programming languages does he know?"
        ]
        
        print(f"\nTesting {len(test_queries)} queries:")
        print("-" * 50)
        
        success_count = 0
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: {query}")
            try:
                response = chatbot.chat(query)
                if response and len(response) > 10:
                    print(f"   ✅ Success: {response[:150]}...")
                    success_count += 1
                else:
                    print(f"   ⚠️  Empty/short response: {response}")
            except Exception as e:
                print(f"   ❌ Query failed: {e}")
        
        print(f"\n🎯 Results: {success_count}/{len(test_queries)} queries successful ({success_count/len(test_queries)*100:.1f}%)")
        
        if success_count == len(test_queries):
            print("🎉 All tests passed! HuggingFace integration working perfectly!")
        elif success_count >= len(test_queries) * 0.8:
            print("✅ Most tests passed! HuggingFace integration working well!")
        else:
            print("⚠️  Some tests failed. Check the logs above.")
                
    except Exception as e:
        print(f"❌ Chatbot initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    test_huggingface_chatbot()
