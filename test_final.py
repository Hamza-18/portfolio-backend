#!/usr/bin/env python3
"""
Final comprehensive test suite for the chatbot after routing fixes
Tests all query types including the previously failing ones
"""

from chatbot.chatbot_engine import RAGChatbot

def test_chatbot():
    print("🤖 FINAL COMPREHENSIVE CHATBOT TEST")
    print("=" * 60)
    
    chatbot = RAGChatbot()
    
    # Test categories with expected routing
    test_categories = {
        "INTRODUCTION/ABOUT": [
            ("Tell me about Hamza", "ABOUT"),
            ("Who is Hamza?", "ABOUT"), 
            ("Introduction to Hamza", "ABOUT"),
            ("What's Hamza's background?", "ABOUT")
        ],
        
        "EDUCATION": [
            ("What's Hamza's education?", "EDUCATION"),
            ("Tell me about his degree", "EDUCATION"),
            ("Where did he study?", "EDUCATION"),
            ("What's his educational background?", "EDUCATION")
        ],
        
        "PROJECTS": [
            ("Tell me about his projects", "PROJECT"),
            ("What projects has he worked on?", "PROJECT"),
            ("Tell me about AI projects", "PROJECT"),
            ("Tell me about machine learning projects", "PROJECT"),
            ("Show me his portfolio", "PROJECT")
        ],
        
        "EXPERIENCE/WORK": [
            ("What's his work experience?", "EXPERIENCE"),
            ("Tell me about his job", "EXPERIENCE"),
            ("Where has he worked?", "EXPERIENCE"),
            ("Tell me about his work experience", "EXPERIENCE")
        ],
        
        "CONTACT": [
            ("How can I contact him?", "CONTACT"),
            ("What's his email?", "CONTACT"),
            ("Tell me about his linkedin", "CONTACT"),
            ("Tell me about his github", "CONTACT")
        ],
        
        "GREETINGS": [
            ("Hello", "GREETING"),
            ("Hi there", "GREETING"),
            ("Hey", "GREETING"),
            ("Good morning", "GREETING")
        ],
        
        "FAREWELLS": [
            ("Goodbye", "FAREWELL"),
            ("Thank you", "FAREWELL"),
            ("Thanks", "FAREWELL"),
            ("Bye", "FAREWELL")
        ],
        
        "SPECIFIC SKILLS": [
            ("Tell me about his programming skills", "VECTOR_SEARCH"),
            ("Tell me about his technologies", "VECTOR_SEARCH"),
            ("What programming languages does he know?", "VECTOR_SEARCH")
        ]
    }
    
    total_tests = 0
    total_passed = 0
    
    for category, tests in test_categories.items():
        print(f"\n📋 TESTING: {category}")
        print("-" * 50)
        
        category_passed = 0
        
        for query, expected_handler in tests:
            total_tests += 1
            print(f"\nQuery: '{query}'")
            
            try:
                response = chatbot.chat(query)
                
                # Determine actual handler based on response patterns
                actual_handler = "UNKNOWN"
                if "Here's an introduction to Hamza:" in response:
                    actual_handler = "ABOUT"
                elif "Here is Hamza's educational background:" in response:
                    actual_handler = "EDUCATION"
                elif "Here are some of Hamza's key projects:" in response:
                    actual_handler = "PROJECT"
                elif "Here's information about Hamza's professional experience:" in response:
                    actual_handler = "EXPERIENCE"
                elif "You can reach Hamza through:" in response:
                    actual_handler = "CONTACT"
                elif "Hello!" in response or "Hi there!" in response:
                    actual_handler = "GREETING"
                elif "Thank you" in response or "Goodbye" in response:
                    actual_handler = "FAREWELL"
                elif "Based on Hamza's background:" in response:
                    actual_handler = "VECTOR_SEARCH"
                
                # Check if routing is correct
                if actual_handler == expected_handler:
                    print(f"   ✅ PASS: Routed to {actual_handler}")
                    total_passed += 1
                    category_passed += 1
                else:
                    print(f"   ❌ FAIL: Expected {expected_handler}, got {actual_handler}")
                    
            except Exception as e:
                print(f"   💥 ERROR: {str(e)}")
        
        category_success = (category_passed / len(tests)) * 100
        print(f"\n📊 {category} Success Rate: {category_passed}/{len(tests)} ({category_success:.1f}%)")
    
    # Overall results
    overall_success = (total_passed / total_tests) * 100
    print(f"\n🎯 OVERALL TEST RESULTS")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_tests - total_passed}")
    print(f"Success Rate: {overall_success:.1f}%")
    
    # Success threshold
    if overall_success >= 90:
        print("\n🎉 EXCELLENT! Chatbot is working great!")
    elif overall_success >= 80:
        print("\n✅ GOOD! Chatbot is working well with minor issues")
    elif overall_success >= 70:
        print("\n⚠️  ACCEPTABLE: Chatbot needs some improvements")
    else:
        print("\n❌ NEEDS WORK: Chatbot requires significant fixes")
    
    return overall_success

if __name__ == "__main__":
    test_chatbot()
