# src/agents/medical_advisor.py
import json
import re
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

class SimpleMedicalAdvisor:
    """Simple Medical Advisor - Health Guidance for Common Queries"""
    
    def __init__(self):
        self.common_symptoms = {
            'fever': {'severity': 'medium', 'care': 'rest_hydration'},
            'headache': {'severity': 'low', 'care': 'rest_medication'},
            'cough': {'severity': 'low', 'care': 'warm_fluids'},
            'cold': {'severity': 'low', 'care': 'rest_fluids'},
            'sore_throat': {'severity': 'low', 'care': 'warm_salt_water'},
            'nausea': {'severity': 'medium', 'care': 'bland_diet'},
            'diarrhea': {'severity': 'medium', 'care': 'hydration_brat_diet'},
            'chest_pain': {'severity': 'high', 'care': 'seek_immediate_help'},
            'difficulty_breathing': {'severity': 'high', 'care': 'emergency'},
            'severe_abdominal_pain': {'severity': 'high', 'care': 'emergency'}
        }
        
        self.wellness_topics = {
            'diet': 'nutrition_guidance',
            'exercise': 'fitness_guidance', 
            'sleep': 'sleep_guidance',
            'stress': 'stress_management',
            'weight': 'weight_management',
            'mental_health': 'mental_wellness'
        }
        
        print("🏥 Simple Medical Advisor initialized - Ready for health guidance!")
    
    async def provide_health_guidance(self, user_query: str, user_context: Dict = None) -> Dict[str, Any]:
        """Main entry point for health guidance"""
        try:
            # Analyze what user is asking about
            query_analysis = await self.analyze_health_query(user_query)
            
            # Generate appropriate guidance
            guidance = await self.generate_health_response(user_query, query_analysis)
            
            return guidance
            
        except Exception as e:
            return await self.handle_medical_error(e, user_query)
    
    async def analyze_health_query(self, query: str) -> Dict[str, Any]:
        """Simple analysis of health query"""
        query_lower = query.lower()
        
        # Symptom queries
        if any(word in query_lower for word in ['symptom', 'feel', 'pain', 'hurt', 'sick']):
            return {'type': 'symptom_check', 'urgency': self.assess_urgency(query)}
        
        # Wellness and prevention
        elif any(word in query_lower for word in ['diet', 'exercise', 'healthy', 'wellness', 'prevent']):
            return {'type': 'wellness_guidance', 'focus': self.identify_wellness_focus(query)}
        
        # Mental health
        elif any(word in query_lower for word in ['stress', 'anxiety', 'depression', 'mental', 'mood']):
            return {'type': 'mental_health', 'concern': self.identify_mental_concern(query)}
        
        # Medication or treatment
        elif any(word in query_lower for word in ['medicine', 'medication', 'treatment', 'cure']):
            return {'type': 'treatment_info', 'focus': 'general'}
        
        # Emergency situations
        elif any(word in query_lower for word in ['emergency', 'urgent', 'severe', 'intense']):
            return {'type': 'emergency_guidance', 'urgency': 'high'}
        
        else:
            return {'type': 'general_health', 'focus': 'general'}
    
    def assess_urgency(self, query: str) -> str:
        """Simple urgency assessment"""
        query_lower = query.lower()
        
        high_urgency = ['chest pain', 'can\'t breathe', 'severe', 'emergency', 'blood', 'unconscious']
        medium_urgency = ['fever', 'vomiting', 'dizzy', 'intense pain']
        
        if any(term in query_lower for term in high_urgency):
            return 'high'
        elif any(term in query_lower for term in medium_urgency):
            return 'medium'
        else:
            return 'low'
    
    def identify_wellness_focus(self, query: str) -> str:
        """Identify wellness focus area"""
        query_lower = query.lower()
        
        for topic in self.wellness_topics:
            if topic in query_lower:
                return topic
        return 'general_wellness'
    
    def identify_mental_concern(self, query: str) -> str:
        """Identify mental health concern"""
        query_lower = query.lower()
        
        if 'stress' in query_lower:
            return 'stress'
        elif 'anxiety' in query_lower:
            return 'anxiety'
        elif 'depression' in query_lower:
            return 'depression'
        else:
            return 'general_mental_health'
    
    async def generate_health_response(self, query: str, analysis: Dict) -> Dict[str, Any]:
        """Generate appropriate health response"""
        
        response_type = analysis['type']
        
        if response_type == 'symptom_check':
            return await self.provide_symptom_guidance(query, analysis)
        elif response_type == 'wellness_guidance':
            return await self.provide_wellness_guidance(query, analysis)
        elif response_type == 'mental_health':
            return await self.provide_mental_health_guidance(query, analysis)
        elif response_type == 'treatment_info':
            return await self.provide_treatment_info(query, analysis)
        elif response_type == 'emergency_guidance':
            return await self.provide_emergency_guidance(query, analysis)
        else:
            return await self.provide_general_health_info(query, analysis)
    
    async def provide_symptom_guidance(self, query: str, analysis: Dict) -> Dict[str, Any]:
        """Simple symptom guidance"""
        
        urgency = analysis.get('urgency', 'low')
        
        if urgency == 'high':
            content = """🚨 **URGENT MEDICAL ATTENTION NEEDED**

**Seek immediate medical help if experiencing:**
- Chest pain or pressure
- Difficulty breathing or shortness of breath
- Severe abdominal pain
- Signs of stroke (face drooping, arm weakness, speech difficulty)
- Loss of consciousness
- Severe bleeding
- High fever with stiff neck

**What to do RIGHT NOW:**
1. Call emergency services (911/local emergency number)
2. Don't drive yourself - get someone else to drive or call ambulance
3. Stay calm and follow dispatcher instructions
4. Have someone stay with you if possible

**This is not a substitute for emergency medical care. Get professional help immediately.**"""

        elif urgency == 'medium':
            content = """⚠️ **MODERATE SYMPTOMS - MONITOR CLOSELY**

**Common moderate symptoms guidance:**

**For Fever:**
- Rest and stay hydrated
- Take fever reducer (acetaminophen/ibuprofen) as directed
- Monitor temperature regularly
- Seek care if fever >101.3°F (38.5°C) for more than 3 days

**For Persistent Symptoms:**
- Keep track of when symptoms started
- Note what makes them better or worse
- Contact healthcare provider if symptoms worsen
- Consider telemedicine consultation

**When to seek immediate care:**
- Symptoms suddenly become severe
- New concerning symptoms develop
- You feel something is seriously wrong

**Remember:** Trust your instincts. If you're worried, contact a healthcare professional."""

        else:
            content = """💡 **MILD SYMPTOMS - SELF-CARE GUIDANCE**

**Common mild symptom management:**

**For Headaches:**
- Rest in a quiet, dark room
- Stay hydrated
- Apply cold or warm compress
- Over-the-counter pain relief if needed

**For Cold/Cough:**
- Get plenty of rest
- Drink warm fluids (tea, broth, warm water)
- Use humidifier or breathe steam
- Gargle with salt water for sore throat

**For Minor Aches:**
- Gentle stretching or movement
- Apply heat or ice as comfortable
- Stay hydrated
- Get adequate sleep

**General Self-Care:**
- Listen to your body
- Rest when you need it
- Maintain good nutrition
- Stay hydrated

**Contact healthcare provider if symptoms persist >1 week or worsen.**"""

        return {
            'success': True,
            'guidance_type': f'Symptom Guidance - {urgency.title()} Urgency',
            'content': content,
            'urgency_level': urgency,
            'next_steps': self.get_symptom_next_steps(urgency),
            'disclaimer': 'This is general health information, not personalized medical advice. Consult healthcare professionals for proper diagnosis and treatment.'
        }
    
    async def provide_wellness_guidance(self, query: str, analysis: Dict) -> Dict[str, Any]:
        """Simple wellness guidance"""
        
        focus = analysis.get('focus', 'general_wellness')
        
        if focus == 'diet':
            content = """🥗 **SIMPLE NUTRITION GUIDANCE**

**Healthy Eating Basics:**
- Eat plenty of fruits and vegetables (aim for 5-9 servings daily)
- Choose whole grains over refined grains
- Include lean proteins (fish, poultry, beans, nuts)
- Limit processed foods and added sugars
- Stay hydrated (8 glasses of water daily)

**Easy Meal Planning:**
- Fill half your plate with vegetables
- Quarter with lean protein
- Quarter with whole grains
- Add a piece of fruit

**Healthy Snacks:**
- Fresh fruit with nuts
- Vegetables with hummus
- Greek yogurt with berries
- Whole grain crackers with cheese

**Simple Tips:**
- Cook at home more often
- Read nutrition labels
- Control portion sizes
- Eat mindfully without distractions"""

        elif focus == 'exercise':
            content = """💪 **SIMPLE FITNESS GUIDANCE**

**Basic Exercise Recommendations:**
- Adults: 150 minutes moderate activity per week
- OR 75 minutes vigorous activity per week
- Plus muscle strengthening 2+ days per week

**Easy Ways to Start:**
- Take 10-minute walks after meals
- Use stairs instead of elevators
- Park farther away from destinations
- Do bodyweight exercises (push-ups, squats)

**Simple Weekly Plan:**
- Monday: 30-minute walk
- Tuesday: Strength exercises (15-20 minutes)
- Wednesday: Active rest (gentle stretching)
- Thursday: 30-minute walk or bike ride
- Friday: Strength exercises
- Weekend: Fun activity (dancing, hiking, sports)

**Getting Started Tips:**
- Start small and build gradually
- Find activities you enjoy
- Exercise with friends for motivation
- Listen to your body and rest when needed"""

        elif focus == 'sleep':
            content = """😴 **SIMPLE SLEEP GUIDANCE**

**Good Sleep Habits:**
- Aim for 7-9 hours per night
- Go to bed and wake up at consistent times
- Create a relaxing bedtime routine
- Keep bedroom cool, dark, and quiet

**Before Bed (1-2 hours):**
- Avoid screens or use blue light filters
- No large meals or caffeine
- Try relaxing activities (reading, gentle stretches)
- Dim the lights

**Sleep Environment:**
- Comfortable mattress and pillows
- Room temperature around 65-68°F (18-20°C)
- Blackout curtains or eye mask
- White noise or earplugs if needed

**If You Can't Sleep:**
- Don't lie awake for hours
- Get up and do quiet activity until sleepy
- Avoid checking the time repeatedly
- Practice deep breathing or meditation"""

        else:
            content = """🌟 **GENERAL WELLNESS GUIDANCE**

**Daily Wellness Habits:**
- Stay hydrated throughout the day
- Eat balanced, nutritious meals
- Get regular physical activity
- Practice good sleep hygiene
- Manage stress through relaxation

**Weekly Wellness Goals:**
- Plan and prep healthy meals
- Schedule regular exercise
- Connect with friends and family
- Spend time in nature
- Practice mindfulness or meditation

**Monthly Health Checks:**
- Review and adjust health goals
- Schedule preventive healthcare appointments
- Assess mental and emotional wellbeing
- Update emergency contacts and health information

**Simple Daily Routine:**
- Morning: Hydrate, healthy breakfast, brief exercise
- Midday: Nutritious lunch, short walk, stress check
- Evening: Balanced dinner, relaxation, good sleep prep"""

        return {
            'success': True,
            'guidance_type': f'Wellness Guidance - {focus.replace("_", " ").title()}',
            'content': content,
            'focus_area': focus,
            'action_items': [
                'Choose 1-2 small changes to implement this week',
                'Track progress with simple daily check-ins',
                'Build habits gradually over time',
                'Celebrate small victories along the way'
            ]
        }
    
    async def provide_mental_health_guidance(self, query: str, analysis: Dict) -> Dict[str, Any]:
        """Simple mental health guidance"""
        
        concern = analysis.get('concern', 'general_mental_health')
        
        content = """🧠 **MENTAL HEALTH SUPPORT GUIDANCE**

**Immediate Support:**
If you're having thoughts of self-harm or suicide:
- Call 988 (Suicide & Crisis Lifeline) - Available 24/7
- Text "HELLO" to 741741 (Crisis Text Line)
- Go to your nearest emergency room
- Call 911

**Daily Mental Health Care:**
- Practice deep breathing (4-7-8 technique)
- Get outside for fresh air and sunlight
- Connect with supportive friends/family
- Engage in activities you enjoy
- Maintain regular sleep schedule

**Stress Management:**
- Identify stress triggers
- Practice mindfulness or meditation
- Use progressive muscle relaxation
- Take breaks throughout the day
- Set realistic expectations

**When to Seek Professional Help:**
- Persistent sadness or anxiety for 2+ weeks
- Difficulty functioning in daily life
- Changes in sleep, appetite, or energy
- Loss of interest in activities you used to enjoy
- Thoughts of self-harm

**Building Resilience:**
- Develop healthy coping strategies
- Build strong social connections
- Practice gratitude daily
- Engage in regular physical activity
- Learn stress management techniques

**Remember:** Mental health is just as important as physical health. It's okay to ask for help, and seeking support is a sign of strength, not weakness."""

        return {
            'success': True,
            'guidance_type': 'Mental Health Support',
            'content': content,
            'concern_area': concern,
            'resources': [
                'National Suicide Prevention Lifeline: 988',
                'Crisis Text Line: Text HOME to 741741',
                'NAMI (National Alliance on Mental Illness): nami.org',
                'Mental Health America: mhanational.org',
                'Psychology Today therapist finder: psychologytoday.com'
            ],
            'disclaimer': 'This is general mental health information. For personalized care, please consult with a mental health professional.'
        }
    
    async def provide_treatment_info(self, query: str, analysis: Dict) -> Dict[str, Any]:
        """Simple treatment information"""
        
        content = """💊 **GENERAL TREATMENT INFORMATION**

**Over-the-Counter Medications:**
- Always read labels and follow dosing instructions
- Check for drug interactions and allergies
- Don't exceed recommended doses
- Consult pharmacist if you have questions

**Common OTC Options:**
- Pain/Fever: Acetaminophen (Tylenol), Ibuprofen (Advil)
- Allergies: Antihistamines (Benadryl, Claritin)
- Cough/Cold: Cough suppressants, decongestants
- Stomach: Antacids, anti-diarrheal medications

**When to See a Healthcare Provider:**
- Symptoms persist or worsen
- You're unsure about self-treatment
- You have chronic health conditions
- You take prescription medications
- Symptoms are severe or concerning

**Home Remedies (Generally Safe):**
- Rest and adequate sleep
- Staying well-hydrated
- Warm salt water gargles for sore throat
- Honey for cough (not for children under 1 year)
- Cold/warm compresses for aches

**Important Reminders:**
- This is general information, not medical advice
- Always consult healthcare providers for diagnosis
- Keep emergency numbers readily available
- Don't delay professional care when needed"""

        return {
            'success': True,
            'guidance_type': 'Treatment Information',
            'content': content,
            'safety_notes': [
                'Always follow medication instructions',
                'Check for allergies and interactions',
                'Consult professionals for persistent symptoms',
                'Keep emergency contacts accessible'
            ],
            'disclaimer': 'This is educational information only. Consult healthcare providers for medical advice and treatment.'
        }
    
    async def provide_emergency_guidance(self, query: str, analysis: Dict) -> Dict[str, Any]:
        """Emergency situation guidance"""
        
        content = """🚨 **EMERGENCY MEDICAL GUIDANCE**

**CALL 911 IMMEDIATELY FOR:**
- Chest pain or pressure
- Difficulty breathing
- Severe bleeding
- Loss of consciousness
- Signs of stroke (F.A.S.T.): Face drooping, Arm weakness, Speech difficulty, Time to call 911
- Severe allergic reactions
- Suspected poisoning
- Major injuries from accidents

**WHILE WAITING FOR HELP:**
- Stay calm and follow dispatcher instructions
- Don't move seriously injured person unless in danger
- Apply pressure to bleeding wounds with clean cloth
- If person is unconscious but breathing, place in recovery position
- Don't give food or water to unconscious person
- Stay with the person and monitor breathing

**IMPORTANT EMERGENCY NUMBERS:**
- Emergency Services: 911
- Poison Control: 1-800-222-1222
- Non-emergency medical advice: Contact your doctor or local urgent care

**BE PREPARED:**
- Keep emergency contacts easily accessible
- Know your address and nearest cross streets
- Have basic first aid supplies at home
- Keep important medical information handy

**REMEMBER:** In true emergencies, don't hesitate to call for professional help. Emergency responders are trained to handle these situations."""

        return {
            'success': True,
            'guidance_type': 'Emergency Medical Guidance',
            'content': content,
            'urgency_level': 'emergency',
            'immediate_action': 'CALL 911 or local emergency services immediately',
            'disclaimer': 'This is emergency guidance information. Always call professional emergency services for serious medical emergencies.'
        }
    
    async def provide_general_health_info(self, query: str, analysis: Dict) -> Dict[str, Any]:
        """General health information"""
        
        content = """🏥 **GENERAL HEALTH INFORMATION**

**Maintaining Good Health:**
- Get regular check-ups with healthcare provider
- Stay up-to-date with vaccinations
- Practice preventive care (screenings, dental care)
- Maintain healthy lifestyle habits
- Know your family medical history

**Daily Health Habits:**
- Eat a balanced diet with variety
- Stay physically active
- Get adequate sleep (7-9 hours)
- Manage stress effectively
- Avoid tobacco and limit alcohol

**When to Contact Healthcare Provider:**
- Annual wellness visits
- Persistent or concerning symptoms
- Changes in health status
- Questions about medications
- Preventive screenings (mammograms, colonoscopies, etc.)

**Building a Health Team:**
- Primary care physician
- Specialists as needed
- Pharmacist for medication questions
- Mental health professional if needed
- Emergency contacts

**Health Resources:**
- Reliable health websites (CDC, Mayo Clinic, WebMD)
- Your healthcare provider's patient portal
- Local health departments
- Community health centers
- Telehealth services

**Remember:** Prevention is often easier and more effective than treatment. Regular healthcare and healthy lifestyle choices are your best investment in long-term wellbeing."""

        return {
            'success': True,
            'guidance_type': 'General Health Information',
            'content': content,
            'health_tips': [
                'Schedule annual check-ups',
                'Keep health records organized',
                'Learn basic first aid',
                'Build healthy daily routines',
                'Stay informed about health topics'
            ],
            'disclaimer': 'This is general health education. For personalized medical advice, consult with qualified healthcare professionals.'
        }
    
    def get_symptom_next_steps(self, urgency: str) -> List[str]:
        """Get next steps based on urgency level"""
        if urgency == 'high':
            return [
                'Seek immediate emergency medical attention',
                'Call 911 or go to emergency room',
                'Don\'t drive yourself if possible',
                'Have someone stay with you'
            ]
        elif urgency == 'medium':
            return [
                'Monitor symptoms closely',
                'Contact healthcare provider if symptoms worsen',
                'Keep track of symptom progression',
                'Consider telemedicine consultation'
            ]
        else:
            return [
                'Try appropriate self-care measures',
                'Get adequate rest and hydration',
                'Monitor symptoms for changes',
                'Contact healthcare provider if symptoms persist >1 week'
            ]
    
    async def handle_medical_error(self, error: Exception, query: str) -> Dict[str, Any]:
        """Handle medical consultation errors"""
        return {
            'success': False,
            'error': 'I encountered an issue processing your health question.',
            'suggestion': 'For medical concerns, please consult with qualified healthcare professionals.',
            'emergency_reminder': 'For medical emergencies, call 911 or your local emergency services immediately.',
            'available_help': [
                'General health information and guidance',
                'Wellness and prevention tips',
                'When to seek medical care guidance',
                'Mental health support resources',
                'Emergency situation guidance'
            ],
            'disclaimer': 'This service provides general health information only, not personalized medical advice.'
        }

# Test the medical advisor
async def test_medical_advisor():
    """Test the medical advisor with various health queries"""
    advisor = SimpleMedicalAdvisor()
    
    test_queries = [
        "I have a headache and feel tired",
        "How can I eat healthier?",
        "I'm feeling stressed and anxious",
        "What should I do for a fever?",
        "I have chest pain and trouble breathing",
        "How much exercise should I get?",
        "I can't sleep well at night",
        "General health tips for staying healthy"
    ]
    
    for query in test_queries:
        print(f"\n🏥 Query: {query}")
        result = await advisor.provide_health_guidance(query)
        if result['success']:
            print(f"   ✅ Guidance Type: {result['guidance_type']}")
            if 'urgency_level' in result:
                print(f"   ⚠️ Urgency: {result['urgency_level']}")
        else:
            print(f"   ❌ Error: {result['error']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_medical_advisor())
