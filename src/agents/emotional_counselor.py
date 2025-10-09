# src/agents/emotional_counselor.py
import asyncio
from typing import Dict, Any

class SimpleEmotionalCounselor:
    """Simple Emotional Support - Easy to Understand & Explain"""
    
    def __init__(self):
        self.emotions = {
            'sad': 'sadness_support',
            'angry': 'anger_management', 
            'anxious': 'anxiety_help',
            'stressed': 'stress_relief',
            'lonely': 'loneliness_support',
            'happy': 'positive_reinforcement'
        }
        print("💙 Simple Emotional Counselor ready!")
    
    async def provide_support(self, user_query: str) -> Dict[str, Any]:
        """Main function - very simple"""
        try:
            # Simple emotion detection
            emotion = self.detect_emotion(user_query)
            
            # Simple supportive response
            response = self.get_support_response(emotion, user_query)
            
            return {
                'success': True,
                'emotion_detected': emotion,
                'support_message': response,
                'helpful_tips': self.get_simple_tips(emotion)
            }
        except:
            return {
                'success': False,
                'message': "I'm here to listen and support you."  # Fixed with double quotes!
            }
    
    def detect_emotion(self, query: str) -> str:
        """Very simple emotion detection"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['sad', 'crying', 'down', 'depressed']):
            return 'sad'
        elif any(word in query_lower for word in ['angry', 'mad', 'frustrated', 'annoyed']):
            return 'angry'
        elif any(word in query_lower for word in ['anxious', 'worried', 'nervous', 'scared']):
            return 'anxious'
        elif any(word in query_lower for word in ['stressed', 'pressure', 'overwhelmed']):
            return 'stressed'
        elif any(word in query_lower for word in ['lonely', 'alone', 'isolated']):
            return 'lonely'
        elif any(word in query_lower for word in ['happy', 'excited', 'good', 'great']):
            return 'happy'
        else:
            return 'general'
    
    def get_support_response(self, emotion: str, query: str) -> str:
        """Simple supportive responses"""
        responses = {
            'sad': "I hear that you're feeling sad. It's okay to feel this way, and your feelings are valid. Take it one step at a time.",
            'angry': "I understand you're feeling angry. Let's take a deep breath together and find healthy ways to process these feelings.",
            'anxious': "Anxiety can feel overwhelming, but you're not alone. Try focusing on what you can control right now.",
            'stressed': "Stress is tough. Remember that it's okay to take breaks and ask for help when you need it.",
            'lonely': "Feeling lonely is hard. You matter, and there are people who care about you, even when it doesn't feel that way.",
            'happy': "I'm glad to hear you're feeling good! It's wonderful to celebrate positive moments.",
            'general': "Thank you for sharing with me. I'm here to listen and support you through whatever you're experiencing."
        }
        return responses.get(emotion, responses['general'])
    
    def get_simple_tips(self, emotion: str) -> list:
        """Simple, practical tips"""
        tips = {
            'sad': ["Take gentle care of yourself", "Reach out to someone you trust", "Do one small thing you enjoy"],
            'angry': ["Count to 10 slowly", "Take deep breaths", "Go for a short walk"],
            'anxious': ["Focus on your breathing", "Name 5 things you can see around you", "Ground yourself in the present"],
            'stressed': ["Take a 5-minute break", "Write down your thoughts", "Talk to someone"],
            'lonely': ["Call a friend or family member", "Join an online community", "Practice self-compassion"],
            'happy': ["Savor this feeling", "Share your joy with others", "Remember this moment"],
            'general': ["Be kind to yourself", "Take things one step at a time", "It's okay to ask for help"]
        }
        return tips.get(emotion, tips['general'])

# Simple test
async def test():
    counselor = SimpleEmotionalCounselor()
    result = await counselor.provide_support("I'm feeling really stressed")
    print(result)

if __name__ == "__main__":
    asyncio.run(test())
