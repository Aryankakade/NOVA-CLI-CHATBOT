# src/agents/career_coach.py
import json
import re
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta

class ProfessionalCareerCoach:
    """Claude.ai Level Career Coach - Dynamic Career Guidance for Any Role/Industry"""
    
    def __init__(self):
        self.career_domains = {
            'technology': {'growth_rate': 0.15, 'avg_salary': 120000, 'demand': 'very_high'},
            'data_science': {'growth_rate': 0.22, 'avg_salary': 130000, 'demand': 'very_high'},
            'digital_marketing': {'growth_rate': 0.10, 'avg_salary': 80000, 'demand': 'high'},
            'product_management': {'growth_rate': 0.18, 'avg_salary': 140000, 'demand': 'very_high'},
            'finance': {'growth_rate': 0.08, 'avg_salary': 110000, 'demand': 'medium'},
            'healthcare': {'growth_rate': 0.12, 'avg_salary': 95000, 'demand': 'high'},
            'education': {'growth_rate': 0.06, 'avg_salary': 65000, 'demand': 'medium'},
            'sales': {'growth_rate': 0.09, 'avg_salary': 85000, 'demand': 'high'},
            'design': {'growth_rate': 0.11, 'avg_salary': 75000, 'demand': 'medium'},
            'consulting': {'growth_rate': 0.07, 'avg_salary': 105000, 'demand': 'medium'}
        }
        print("💼 Professional Career Coach initialized - Ready for ANY career guidance!")
    
    async def provide_career_guidance(self, user_query: str, user_profile: Dict = None) -> Dict[str, Any]:
        """Main entry point - handles ANY career query dynamically"""
        try:
            intent_analysis = await self.analyze_career_intent(user_query, user_profile)
            guidance = await self.generate_dynamic_guidance(user_query, intent_analysis, user_profile)
            return guidance
        except Exception as e:
            return await self.handle_career_error(e, user_query)
    
    async def analyze_career_intent(self, query: str, profile: Dict = None) -> Dict[str, Any]:
        """Dynamically analyze what type of career help user needs"""
        query_lower = query.lower()
        if any(word in query_lower for word in ['resume', 'cv', 'curriculum vitae']):
            return {'primary_intent': 'resume_help', 'confidence': 0.9, 'specifics': self.extract_resume_specifics(query)}
        elif any(word in query_lower for word in ['interview', 'interviewing']):
            return {'primary_intent': 'interview_prep', 'confidence': 0.9, 'specifics': self.extract_interview_specifics(query)}
        elif any(word in query_lower for word in ['salary', 'compensation', 'negotiate', 'pay']):
            return {'primary_intent': 'salary_guidance', 'confidence': 0.9, 'specifics': self.extract_salary_specifics(query)}
        elif any(word in query_lower for word in ['career change', 'transition', 'switch']):
            return {'primary_intent': 'career_transition', 'confidence': 0.9, 'specifics': self.extract_transition_specifics(query)}
        elif any(word in query_lower for word in ['skills', 'learn', 'training', 'course', 'roadmap']):
            return {'primary_intent': 'skill_development', 'confidence': 0.8, 'specifics': self.extract_skill_specifics(query)}
        elif any(word in query_lower for word in ['job search', 'job hunting', 'find job']):
            return {'primary_intent': 'job_search', 'confidence': 0.9, 'specifics': self.extract_job_search_specifics(query)}
        elif any(word in query_lower for word in ['networking', 'connections', 'linkedin']):
            return {'primary_intent': 'networking', 'confidence': 0.8, 'specifics': self.extract_networking_specifics(query)}
        elif any(word in query_lower for word in ['career plan', 'career path', 'future', 'roadmap']):
            return {'primary_intent': 'career_planning', 'confidence': 0.8, 'specifics': self.extract_planning_specifics(query)}
        else:
            return {'primary_intent': 'general_consultation', 'confidence': 0.6, 'specifics': {}}
    
    def extract_resume_specifics(self, query: str) -> Dict:
        """Extract specific resume requirements from query"""
        specifics = {'action': 'general', 'role': None, 'experience_level': None}
        query_lower = query.lower()
        if any(word in query_lower for word in ['create', 'build', 'write']):
            specifics['action'] = 'create'
        elif any(word in query_lower for word in ['improve', 'optimize', 'review']):
            specifics['action'] = 'optimize'
        role_keywords = ['engineer', 'developer', 'analyst', 'manager', 'designer', 'scientist', 'consultant', 'specialist']
        for role in role_keywords:
            if role in query_lower:
                specifics['role'] = role
                break
        if any(word in query_lower for word in ['entry', 'junior', 'fresher', 'graduate']):
            specifics['experience_level'] = 'entry'
        elif any(word in query_lower for word in ['senior', 'lead', 'principal']):
            specifics['experience_level'] = 'senior'
        else:
            specifics['experience_level'] = 'mid'
        return specifics
    
    def extract_interview_specifics(self, query: str) -> Dict:
        """Extract interview preparation specifics"""
        specifics = {'type': 'general', 'role': None, 'company': None}
        query_lower = query.lower()
        if any(word in query_lower for word in ['technical', 'coding', 'programming']):
            specifics['type'] = 'technical'
        elif any(word in query_lower for word in ['behavioral', 'soft skills']):
            specifics['type'] = 'behavioral'
        elif any(word in query_lower for word in ['system design', 'architecture']):
            specifics['type'] = 'system_design'
        companies = ['google', 'microsoft', 'amazon', 'facebook', 'meta', 'apple', 'netflix']
        for company in companies:
            if company in query_lower:
                specifics['company'] = company
                break
        return specifics
    
    def extract_salary_specifics(self, query: str) -> Dict:
        """Extract salary negotiation specifics"""
        specifics = {'stage': 'negotiation', 'role': None, 'experience': None}
        query_lower = query.lower()
        if any(word in query_lower for word in ['research', 'benchmark', 'market']):
            specifics['stage'] = 'research'
        elif any(word in query_lower for word in ['negotiate', 'negotiation']):
            specifics['stage'] = 'negotiation'
        return specifics
    
    def extract_transition_specifics(self, query: str) -> Dict:
        """Extract career transition specifics"""
        specifics = {'from_field': None, 'to_field': None, 'timeline': None}
        query_lower = query.lower()
        transition_patterns = ['from', 'to', 'into', 'switch to', 'move to']
        for pattern in transition_patterns:
            if pattern in query_lower:
                parts = query_lower.split(pattern)
                if len(parts) >= 2:
                    specifics['from_field'] = parts[0].strip()
                    specifics['to_field'] = parts[1].strip()
        return specifics
    
    def extract_skill_specifics(self, query: str) -> Dict:
        """Extract skill development specifics"""
        specifics = {'domain': None, 'level': None, 'timeline': None}
        query_lower = query.lower()
        domains = ['programming', 'data science', 'marketing', 'design', 'management', 'ai', 'cloud', 'cybersecurity']
        for domain in domains:
            if domain in query_lower:
                specifics['domain'] = domain
                break
        if any(word in query_lower for word in ['beginner', 'basic', 'start']):
            specifics['level'] = 'beginner'
        elif any(word in query_lower for word in ['advanced', 'expert', 'master']):
            specifics['level'] = 'advanced'
        else:
            specifics['level'] = 'intermediate'
        return specifics
    
    def extract_job_search_specifics(self, query: str) -> Dict:
        """Extract job search specifics"""
        return {'industry': None, 'remote': 'remote' in query.lower(), 'location': None}
    
    def extract_networking_specifics(self, query: str) -> Dict:
        """Extract networking specifics"""
        return {'platform': 'linkedin' if 'linkedin' in query.lower() else 'general', 'industry': None}
    
    def extract_planning_specifics(self, query: str) -> Dict:
        """Extract career planning specifics"""
        return {'timeline': '5 years' if '5' in query else '3 years', 'industry': None}
    
    async def generate_dynamic_guidance(self, query: str, intent_analysis: Dict, profile: Dict) -> Dict[str, Any]:
        """Generate dynamic guidance based on user's specific query"""
        primary_intent = intent_analysis['primary_intent']
        specifics = intent_analysis.get('specifics', {})
        if primary_intent == 'resume_help':
            return await self.generate_resume_guidance(query, specifics)
        elif primary_intent == 'interview_prep':
            return await self.generate_interview_guidance(query, specifics)
        elif primary_intent == 'salary_guidance':
            return await self.generate_salary_guidance(query, specifics)
        elif primary_intent == 'career_transition':
            return await self.generate_transition_guidance(query, specifics)
        elif primary_intent == 'skill_development':
            return await self.generate_skill_guidance(query, specifics)
        elif primary_intent == 'job_search':
            return await self.generate_job_search_guidance(query, specifics)
        elif primary_intent == 'networking':
            return await self.generate_networking_guidance(query, specifics)
        elif primary_intent == 'career_planning':
            return await self.generate_planning_guidance(query, specifics)
        else:
            return await self.generate_general_guidance(query, specifics)
    
    async def generate_resume_guidance(self, query: str, specifics: Dict) -> Dict[str, Any]:
        """Dynamic resume guidance based on user's specific needs"""
        action = specifics.get('action', 'general')
        role = specifics.get('role', 'any role')
        experience = specifics.get('experience_level', 'mid-level')
        content = f"Professional resume {action} guidance for {role} at {experience} level. Research target job descriptions, use ATS-friendly formatting, quantify achievements with metrics, tailor keywords to each application, and ensure error-free presentation."
        return {'success': True, 'guidance_type': f'Resume {action.title()} - {role.title()} ({experience.title()} Level)', 'content': content, 'action_items': [f'Research 5-10 job descriptions for {role} positions', f'List 10+ quantified achievements relevant to {role}', 'Choose appropriate resume format based on career history', 'Create tailored versions for different applications', 'Get feedback from 2-3 professionals in target industry'], 'timeline': '3-5 days for thorough creation/optimization', 'confidence_score': 0.9}
    
    async def generate_interview_guidance(self, query: str, specifics: Dict) -> Dict[str, Any]:
        """Dynamic interview guidance based on user's needs"""
        interview_type = specifics.get('type', 'general')
        role = specifics.get('role', 'your target role')
        company = specifics.get('company', 'your target company')
        content = f"Comprehensive {interview_type} interview preparation guide. Practice problem-solving frameworks, prepare STAR method stories, research company culture and values, practice with mock interviews, and develop thoughtful questions to ask interviewers."
        return {'success': True, 'guidance_type': f'{interview_type.title()} Interview Preparation', 'content': content, 'action_items': [f'Prepare 5-7 STAR method stories relevant to {role} positions', f'Research {company} thoroughly including recent news and culture', 'Complete 10-15 practice problems on coding platforms', 'Schedule 2-3 mock interviews with peers or professionals', 'Prepare 8-10 thoughtful questions to ask interviewers'], 'timeline': '2-4 weeks for comprehensive preparation', 'confidence_score': 0.9}
    
    async def generate_salary_guidance(self, query: str, specifics: Dict) -> Dict[str, Any]:
        """Dynamic salary guidance based on user's situation"""
        stage = specifics.get('stage', 'negotiation')
        role = specifics.get('role', 'your position')
        content = f"Strategic salary negotiation masterclass. Research market rates from multiple sources, document quantified achievements, practice negotiation conversations, consider total compensation package beyond base salary, and maintain professional collaborative approach throughout process."
        return {'success': True, 'guidance_type': 'Salary Negotiation Strategy', 'content': content, 'action_items': ['Research salary ranges from 3-5 reliable sources', 'Document 8-10 quantified professional achievements', 'Practice negotiation conversation with trusted friend/mentor', 'Prepare 3-5 alternative compensation structures', 'Set target, ideal, and minimum acceptable salary levels'], 'timeline': '2-3 weeks for thorough preparation', 'confidence_score': 0.9}
    
    async def generate_transition_guidance(self, query: str, specifics: Dict) -> Dict[str, Any]:
        """Dynamic career transition guidance"""
        from_field = specifics.get('from_field', 'current field')
        to_field = specifics.get('to_field', 'target field')
        content = f"Strategic career transition roadmap from {from_field} to {to_field}. Conduct skills gap analysis, develop transition timeline, build relevant experience through projects, network in target industry, and create compelling narrative for career change."
        return {'success': True, 'guidance_type': 'Career Transition Strategy', 'content': content, 'action_items': ['Complete comprehensive skills gap analysis', 'Create 12-month transition timeline with milestones', 'Identify and connect with 5 professionals in target field', 'Build 2-3 portfolio projects demonstrating new skills', 'Update LinkedIn and resume to reflect transition narrative'], 'timeline': '6-12 months for complete transition', 'confidence_score': 0.8}
    
    async def generate_skill_guidance(self, query: str, specifics: Dict) -> Dict[str, Any]:
        """Dynamic skill development guidance"""
        domain = specifics.get('domain', 'professional skills')
        level = specifics.get('level', 'intermediate')
        content = f"Comprehensive {level} skill development plan for {domain}. Identify high-impact skills, create structured learning path, build practical projects, seek mentorship opportunities, and track progress with measurable goals."
        return {'success': True, 'guidance_type': f'Skill Development - {domain.title()} ({level.title()} Level)', 'content': content, 'action_items': [f'Research top 5-7 in-demand skills for {domain}', f'Enroll in {level}-appropriate courses or certifications', 'Build 2-3 hands-on projects demonstrating new skills', 'Find mentor or study group in chosen domain', 'Set monthly learning goals with progress tracking'], 'timeline': '3-6 months for significant skill development', 'confidence_score': 0.8}
    
    async def generate_job_search_guidance(self, query: str, specifics: Dict) -> Dict[str, Any]:
        """Dynamic job search strategy"""
        industry = specifics.get('industry', 'your target industry')
        remote = specifics.get('remote', False)
        content = f"Strategic job search plan for {industry} positions{'including remote opportunities' if remote else ''}. Optimize LinkedIn profile, identify target companies, leverage networking connections, track applications systematically, and follow up professionally."
        return {'success': True, 'guidance_type': 'Job Search Strategy', 'content': content, 'action_items': ['Create list of 50+ target companies and roles', 'Optimize LinkedIn profile with industry keywords', 'Set up job alerts on 3-5 major platforms', 'Reach out to 5 professionals per week for networking', 'Track applications in spreadsheet with follow-up dates'], 'timeline': '3-6 months for comprehensive job search', 'confidence_score': 0.8}
    
    async def generate_networking_guidance(self, query: str, specifics: Dict) -> Dict[str, Any]:
        """Dynamic networking strategy"""
        platform = specifics.get('platform', 'general')
        industry = specifics.get('industry', 'your industry')
        content = f"Professional networking strategy for {industry} using {platform} and other channels. Build authentic relationships, provide value before asking for help, attend industry events, maintain consistent communication, and leverage connections for career opportunities."
        return {'success': True, 'guidance_type': 'Professional Networking Strategy', 'content': content, 'action_items': [f'Optimize {platform} profile with professional photo and compelling headline', 'Identify 20-30 key professionals to connect with in target industry', 'Join 3-5 relevant professional groups or associations', 'Attend 2 industry events or meetups per month', 'Send personalized follow-up messages within 24 hours of meeting'], 'timeline': '2-3 months to build strong network foundation', 'confidence_score': 0.8}
    
    async def generate_planning_guidance(self, query: str, specifics: Dict) -> Dict[str, Any]:
        """Dynamic career planning guidance"""
        timeline = specifics.get('timeline', '5 years')
        industry = specifics.get('industry', 'your industry')
        content = f"Strategic {timeline} career planning roadmap for {industry}. Define career goals and values, assess current position, identify growth opportunities, develop required skills, and create actionable milestones with regular progress reviews."
        return {'success': True, 'guidance_type': f'{timeline.title()} Career Planning Strategy', 'content': content, 'action_items': [f'Define specific career goals for next {timeline}', 'Conduct honest assessment of current skills and experience', 'Research career progression paths in target industry', 'Create annual milestones with quarterly check-ins', 'Identify 3-5 key skills to develop each year'], 'timeline': f'{timeline} with annual reviews and adjustments', 'confidence_score': 0.8}
    
    async def generate_general_guidance(self, query: str, specifics: Dict) -> Dict[str, Any]:
        """General career consultation"""
        content = "Comprehensive career guidance covering resume optimization, interview preparation, salary negotiation, skill development, job search strategies, professional networking, and career planning. Tailored approach based on your specific goals, industry, and experience level."
        return {'success': True, 'guidance_type': 'General Career Consultation', 'content': content, 'action_items': ['Clarify specific career goals and priorities', 'Assess current position and desired direction', 'Identify 2-3 most important areas for improvement', 'Create 90-day action plan with specific milestones', 'Schedule regular progress reviews and adjustments'], 'timeline': 'Ongoing with regular milestone reviews', 'confidence_score': 0.7}
    
    def assess_transition_difficulty(self, from_field: str, to_field: str) -> str:
        """Assess difficulty level of career transition"""
        if not from_field or not to_field:
            return "Medium - depends on transferable skills"
        return "Medium to High - requires strategic planning"
    
    def estimate_transition_timeline(self, from_field: str, to_field: str) -> str:
        """Estimate realistic timeline for transition"""
        return "6-18 months depending on skill gap and market conditions"
    
    def get_transition_skills(self, from_field: str, to_field: str) -> str:
        """Get relevant skills for transition"""
        return "Focus on transferable skills and target-specific competencies through courses, projects, and hands-on experience"
    
    def get_role_specific_skills(self, role: str) -> str:
        """Get role-specific skills"""
        return f"Research top skills for {role} positions in current job market"
    
    def get_role_action_verbs(self, role: str) -> str:
        """Get action verbs for role"""
        return f"Use action verbs relevant to {role} responsibilities and achievements"
    
    def get_role_metrics(self, role: str) -> str:
        """Get relevant metrics for role"""
        return f"Include quantifiable results that matter for {role} positions"
    
    def get_role_keywords(self, role: str) -> str:
        """Get keywords for role"""
        return f"Research and include keywords commonly found in {role} job descriptions"
    
    def get_experience_level_tips(self, level: str) -> str:
        """Get tips based on experience level"""
        return f"Tailor resume approach for {level} professionals focusing on relevant strengths and addressing potential concerns"
    
    def get_company_interview_info(self, company: str) -> str:
        """Get company-specific interview information"""
        return f"Research {company}'s interview process, culture, and recent developments to demonstrate genuine interest and preparation"
    
    async def handle_career_error(self, error: Exception, query: str) -> Dict[str, Any]:
        """Handle errors gracefully"""
        return {'success': False, 'error': 'I encountered an issue processing your career question.', 'suggestion': 'Please rephrase your question or provide more specific details about what career help you need.', 'available_help': ['Resume optimization and creation', 'Interview preparation (technical and behavioral)', 'Salary negotiation strategies', 'Career transition planning', 'Skill development recommendations', 'Job search strategies', 'Professional networking guidance', 'Career planning and goal setting']}

async def test_career_coach():
    """Test the professional career coach"""
    coach = ProfessionalCareerCoach()
    test_queries = ["Help me optimize my resume for software engineering roles", "How do I prepare for technical interviews at Google?", "I want to negotiate my salary for a data scientist position", "Guide me through transitioning from marketing to data science", "What skills should I learn to become a product manager?", "How do I build a strong professional network in tech?", "Help me create a 5-year career plan"]
    for query in test_queries:
        print(f"💼 Query: {query}")
        result = await coach.provide_career_guidance(query)
        if result['success']:
            print(f"   ✅ Guidance Type: {result['guidance_type']}")
            print(f"   📋 Action Items: {len(result['action_items'])} provided")
        else:
            print(f"   ❌ Error: {result['error']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_career_coach())
