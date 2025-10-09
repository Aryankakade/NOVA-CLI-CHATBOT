import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
import asyncio # For async operations
from typing import List
import sqlite3
from datetime import timedelta

class SmartExportSystem:
    """Smart conversation history export with multiple formats and filtering"""
    
    def __init__(self, memory_system):
        self.memory_system = memory_system
        self.export_formats = ['json', 'markdown', 'html', 'csv', 'txt']
        print("âœ… Smart Export System initialized")
    
    async def export_conversation_history(self, user_id: str, export_format: str = 'markdown',
                                        filter_options: Dict = None) -> Dict[str, Any]:
        """Export conversation history with smart filtering and formatting"""
        
        try:
            # Get conversation data
            conversations = await self._get_filtered_conversations(user_id, filter_options or {})
            
            if not conversations:
                return {"error": "No conversations found to export"}
            
            # Generate export based on format
            if export_format.lower() == 'markdown':
                content = self._export_to_markdown(conversations, user_id)
                filename = f"nova_conversations_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            
            elif export_format.lower() == 'json':
                content = self._export_to_json(conversations, user_id)
                filename = f"nova_conversations_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            elif export_format.lower() == 'html':
                content = self._export_to_html(conversations, user_id)
                filename = f"nova_conversations_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            
            elif export_format.lower() == 'csv':
                content = self._export_to_csv(conversations, user_id)
                filename = f"nova_conversations_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            elif export_format.lower() == 'txt':
                content = self._export_to_text(conversations, user_id)
                filename = f"nova_conversations_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            else:
                return {"error": f"Unsupported export format: {export_format}"}
            
            # Save to file
            export_path = os.path.join(os.getcwd(), 'exports')
            os.makedirs(export_path, exist_ok=True)
            
            file_path = os.path.join(export_path, filename)
            
            if export_format.lower() == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            # Generate statistics
            stats = self._generate_export_stats(conversations)
            
            return {
                "success": True,
                "file_path": file_path,
                "filename": filename,
                "format": export_format,
                "stats": stats,
                "file_size": os.path.getsize(file_path),
                "export_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Export failed: {str(e)}"}
    
    async def _get_filtered_conversations(self, user_id: str, filter_options: Dict) -> List[Dict]:
        """Get conversations with smart filtering"""
        
        try:
            # Get all conversations from database
            with sqlite3.connect(self.memory_system.db_path) as conn:
                cursor = conn.cursor()
                
                # Build dynamic query based on filters
                query = '''
                SELECT user_input, bot_response, agent_type, language, emotion, 
                       timestamp, response_time, voice_used, satisfaction_rating,
                       learned_facts, context_summary
                FROM conversations
                WHERE user_id = ?
                '''
                params = [user_id]
                
                # Apply filters
                if filter_options.get('agent_type'):
                    query += ' AND agent_type = ?'
                    params.append(filter_options['agent_type'])
                
                if filter_options.get('language'):
                    query += ' AND language = ?'
                    params.append(filter_options['language'])
                
                if filter_options.get('emotion'):
                    query += ' AND emotion = ?'
                    params.append(filter_options['emotion'])
                
                if filter_options.get('date_from'):
                    query += ' AND timestamp >= ?'
                    params.append(filter_options['date_from'])
                
                if filter_options.get('date_to'):
                    query += ' AND timestamp <= ?'
                    params.append(filter_options['date_to'])
                
                if filter_options.get('voice_only'):
                    query += ' AND voice_used = 1'
                
                if filter_options.get('min_rating'):
                    query += ' AND satisfaction_rating >= ?'
                    params.append(filter_options['min_rating'])
                
                # Order by timestamp
                query += ' ORDER BY timestamp DESC'
                
                # Limit results if specified
                if filter_options.get('limit'):
                    query += ' LIMIT ?'
                    params.append(filter_options['limit'])
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to list of dictionaries
                conversations = []
                columns = ['user_input', 'bot_response', 'agent_type', 'language', 'emotion',
                          'timestamp', 'response_time', 'voice_used', 'satisfaction_rating',
                          'learned_facts', 'context_summary']
                
                for row in rows:
                    conv_dict = dict(zip(columns, row))
                    conversations.append(conv_dict)
                
                return conversations
                
        except Exception as e:
            print(f"Error getting filtered conversations: {e}")
            return []
    
    def _export_to_markdown(self, conversations: List[Dict], user_id: str) -> str:
        """Export to beautifully formatted Markdown"""
        
        content = f"""# ðŸ¤– NOVA AI Conversation History
        
**User:** {user_id}  
**Export Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Total Conversations:** {len(conversations)}  

---

"""
        
        for i, conv in enumerate(conversations, 1):
            timestamp = datetime.fromisoformat(conv['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            agent_emoji = self._get_agent_emoji(conv['agent_type'])
            
            content += f"""## {i}. Conversation #{i}

**ðŸ"… Time:** {timestamp}  
**{agent_emoji} Agent:** {conv['agent_type'].title()}  
**ðŸŒ Language:** {conv['language']}  
**ðŸ˜Š Emotion:** {conv['emotion']}  
**â±ï¸ Response Time:** {conv['response_time']:.2f}s  
"""
            
            if conv['voice_used']:
                content += "**ðŸŽ¤ Voice:** Used  \n"
            
            content += f"""
### ðŸ'¤ User Question:
{conv['user_input']}

### ðŸ¤– NOVA Response:
{conv['bot_response']}

"""
            
            if conv['learned_facts']:
                content += f"""### ðŸ§  What NOVA Learned:
{conv['learned_facts']}

"""
            
            content += "---\n\n"
        
        return content
    
    def _export_to_html(self, conversations: List[Dict], user_id: str) -> str:
        """Export to beautiful HTML with modern styling"""

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NOVA AI Conversations - {user_id}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #667eea;
        }}
        .conversation {{
            background: #f8f9ff;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 25px;
            border-left: 5px solid #667eea;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .user-message {{
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 15px 20px;
            border-radius: 20px 20px 20px 5px;
            margin-bottom: 15px;
        }}
        .nova-response {{
            background: white;
            border: 2px solid #e1e8ff;
            padding: 15px 20px;
            border-radius: 5px 20px 20px 20px;
            margin-top: 15px;
        }}
        .metadata {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 20px;
            font-size: 0.9em;
            color: #666;
        }}
        .metadata span {{
            background: #e1e8ff;
            padding: 5px 12px;
            border-radius: 20px;
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .learned-facts {{
            background: #f0f8ff;
            border-left: 4px solid #4CAF50;
            padding: 15px;
            margin-top: 15px;
            border-radius: 0 10px 10px 0;
        }}
        h1 {{ color: #667eea; margin-bottom: 10px; }}
        h2 {{ color: #764ba2; }}
        .stats {{ 
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>ðŸ¤– NOVA AI Conversation History</h1>
            <p><strong>User:</strong> {user_id}</p>
            <p><strong>Export Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="stats">
            <h2>ðŸ"Š Export Statistics</h2>
            <p><strong>Total Conversations:</strong> {len(conversations)}</p>
        </div>
"""

        for i, conv in enumerate(conversations, 1):
            timestamp = datetime.fromisoformat(conv['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            agent_emoji = self._get_agent_emoji(conv['agent_type'])

            user_input_html = conv['user_input'].replace("\n", "<br>")
            bot_response_html = conv['bot_response'].replace("\n", "<br>")
            learned_facts_html = conv['learned_facts'].replace("\n", "<br>") if conv['learned_facts'] else ""

            html_content += f"""
        <div class="conversation">
            <h2>ðŸ'¬ Conversation #{i}</h2>
            <div class="metadata">
                <span>ðŸ"… {timestamp}</span>
                <span>{agent_emoji} {conv['agent_type'].title()}</span>
                <span>ðŸŒ {conv['language']}</span>
                <span>ðŸ˜Š {conv['emotion']}</span>
                <span>â±ï¸ {conv['response_time']:.2f}s</span>"""

            if conv['voice_used']:
                html_content += '<span>ðŸŽ¤ Voice</span>'

            html_content += f"""
            </div>
            <div class="user-message">
                <strong>ðŸ'¤ You:</strong><br>
                {user_input_html}
            </div>
            <div class="nova-response">
                <strong>ðŸ¤– NOVA:</strong><br>
                {bot_response_html}
            </div>"""

            if learned_facts_html:
                html_content += f"""
            <div class="learned-facts">
                <strong>ðŸ§  What NOVA Learned:</strong><br>
                {learned_facts_html}
            </div>"""

            html_content += "</div>\n"

        html_content += """
    </div>
</body>
</html>"""

        return html_content
    
    def _export_to_csv(self, conversations: List[Dict], user_id: str) -> str:
        """Export to CSV format"""
        
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'Conversation_ID', 'Timestamp', 'User_Input', 'NOVA_Response',
            'Agent_Type', 'Language', 'Emotion', 'Response_Time_Seconds',
            'Voice_Used', 'Learned_Facts', 'Context_Summary'
        ])
        
        # Write data
        for i, conv in enumerate(conversations, 1):
            writer.writerow([
                i,
                conv['timestamp'],
                conv['user_input'].replace('\n', ' ').replace('\r', ' '),
                conv['bot_response'].replace('\n', ' ').replace('\r', ' '),
                conv['agent_type'],
                conv['language'],
                conv['emotion'],
                conv['response_time'],
                'Yes' if conv['voice_used'] else 'No',
                conv['learned_facts'].replace('\n', ' ').replace('\r', ' ') if conv['learned_facts'] else '',
                conv['context_summary'].replace('\n', ' ').replace('\r', ' ') if conv['context_summary'] else ''
            ])
        
        return output.getvalue()
    
    def _export_to_text(self, conversations: List[Dict], user_id: str) -> str:
        """Export to simple text format"""
        
        content = f"""NOVA AI Conversation History
========================

User: {user_id}
Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Conversations: {len(conversations)}

{'='*60}

"""
        
        for i, conv in enumerate(conversations, 1):
            timestamp = datetime.fromisoformat(conv['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            
            content += f"""Conversation #{i}
Time: {timestamp}
Agent: {conv['agent_type'].title()}
Language: {conv['language']}
Emotion: {conv['emotion']}
Response Time: {conv['response_time']:.2f}s
Voice Used: {'Yes' if conv['voice_used'] else 'No'}

USER: {conv['user_input']}

NOVA: {conv['bot_response']}

"""
            if conv['learned_facts']:
                content += f"LEARNED: {conv['learned_facts']}\n\n"
            
            content += f"{'-'*60}\n\n"
        
        return content
    
    def _get_agent_emoji(self, agent_type: str) -> str:
        """Get emoji for agent type"""
        emoji_map = {
            'coding': '🔧',
            'career': '💼',
            'business': '📈',
            'medical': '🏥',
            'emotional': '💙',
            'technical_architect': '🚀',
            'general': '🤖'
        }
        return emoji_map.get(agent_type, '🤖')
    
    def _generate_export_stats(self, conversations: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive statistics for export"""
        
        if not conversations:
            return {}
        
        # Basic counts
        total_conversations = len(conversations)
        voice_conversations = sum(1 for c in conversations if c['voice_used'])
        
        # Agent usage
        agent_counts = {}
        for conv in conversations:
            agent = conv['agent_type']
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        # Language usage
        language_counts = {}
        for conv in conversations:
            lang = conv['language']
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        # Emotion analysis
        emotion_counts = {}
        for conv in conversations:
            emotion = conv['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Response times
        response_times = [float(c['response_time']) for c in conversations if c['response_time']]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Time span
        timestamps = [datetime.fromisoformat(c['timestamp']) for c in conversations]
        time_span = max(timestamps) - min(timestamps) if len(timestamps) > 1 else timedelta(0)
        
        return {
            'total_conversations': total_conversations,
            'voice_conversations': voice_conversations,
            'voice_percentage': f"{(voice_conversations/total_conversations)*100:.1f}%",
            'agent_usage': agent_counts,
            'language_usage': language_counts,
            'emotion_distribution': emotion_counts,
            'average_response_time': f"{avg_response_time:.2f}s",
            'time_span': str(time_span),
            'date_range': {
                'first': min(timestamps).isoformat() if timestamps else None,
                'last': max(timestamps).isoformat() if timestamps else None
            }
        }
    
    def get_export_options(self) -> Dict[str, Any]:
        """Get available export options and filters"""
        return {
            'formats': self.export_formats,
            'filters': {
                'agent_type': ['coding', 'career', 'business', 'medical', 'emotional', 'technical_architect', 'general'],
                'language': ['english', 'hinglish', 'hindi'],
                'emotion': ['neutral', 'excited', 'frustrated', 'sad', 'anxious', 'confident', 'confused'],
                'date_range': 'YYYY-MM-DD format',
                'voice_only': 'boolean',
                'min_rating': '1-5 scale',
                'limit': 'number of conversations'
            },
            'sample_filter': {
                'agent_type': 'coding',
                'language': 'english',
                'date_from': '2024-01-01',
                'limit': 100
            }
        }