"""
qa_engine.py - Enhanced Question-Answering Engine
Debugged version with modern LangChain patterns and comprehensive error handling
Updated to use OpenRouter instead of OpenAI
"""

import os
import sys
import time
import json
import warnings
from typing import Optional, Dict, List, Any, Tuple, Union
from datetime import datetime
import logging
import requests # type: ignore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

try:
    # FIXED: Updated imports for LangChain 0.2+
    from langchain_community.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, AIMessage, SystemMessage
    from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
    from langchain.agents import initialize_agent, Tool, AgentType
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain.prompts import PromptTemplate, ChatPromptTemplate
    from langchain.chains import LLMChain, ConversationChain
    from langchain_community.utilities import GoogleSearchAPIWrapper, WikipediaAPIWrapper
    from langchain_community.tools import DuckDuckGoSearchRun
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    print(f"⚠️ LangChain not available: {e}")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️ Requests library not available")

class QAEngineCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for QA Engine"""
    
    def __init__(self):
        self.start_time = None
        self.tokens_used = 0
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        self.start_time = time.time()
        print("🧠 Processing your question...")
    
    def on_llm_end(self, response, **kwargs) -> None:
        if self.start_time:
            duration = time.time() - self.start_time
            print(f"✅ Response generated in {duration:.2f}s")
    
    def on_llm_error(self, error, **kwargs) -> None:
        print(f"❌ LLM Error: {error}")
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        tool_name = serialized.get("name", "Unknown")
        print(f"🔧 Using tool: {tool_name}")
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        print("🔧 Tool execution completed")

class EnhancedQAEngine:
    def __init__(self,
                 model_name: str = "anthropic/claude-3.5-sonnet",
                 temperature: float = 0.7,
                 max_tokens: int = 500,
                 enable_memory: bool = True,
                 enable_tools: bool = True,
                 memory_type: str = "buffer",
                 max_memory_tokens: int = 1000):
        """
        Initialize the Enhanced QA Engine with OpenRouter API
        
        Args:
            model_name: OpenRouter model name (default: anthropic/claude-3.5-sonnet)
            temperature: Response creativity (0.0 to 1.0)
            max_tokens: Maximum response length
            enable_memory: Enable conversation memory
            enable_tools: Enable external tools
            memory_type: Type of memory ("buffer" or "summary")
            max_memory_tokens: Maximum tokens for memory
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_memory = enable_memory
        self.enable_tools = enable_tools
        self.memory_type = memory_type
        self.max_memory_tokens = max_memory_tokens
        
        # Get OpenRouter API key from environment
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        
        # Initialize components
        self.llm = None
        self.chat_llm = None
        self.memory = None
        self.tools = []
        self.agent = None
        self.conversation_history = []
        self.system_prompt = "You are EVA, a helpful and intelligent AI assistant. Provide clear, accurate, and emotionally aware responses."
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize all components
        self._initialize_llm()
        if self.enable_memory and self.llm:
            self._initialize_memory()
        if self.enable_tools:
            self._initialize_tools()
        if LANGCHAIN_AVAILABLE and self.chat_llm and self.tools:
            self._initialize_agent()

    def _initialize_llm(self):
        """Initialize ChatOpenAI with OpenRouter - FIXED VERSION"""
        try:
            api_key = self.openrouter_api_key
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY not found in environment variables")

            # Initialize callback handler
            self.callback_handler = QAEngineCallbackHandler()

            # FIXED: Use proper ChatOpenAI with OpenRouter
            self.chat_llm = ChatOpenAI(
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
)
            
            # Keep both references for compatibility
            self.llm = self.chat_llm

            print(f"✅ LLM initialized: {self.model_name} (via OpenRouter)")
        except Exception as e:
            print(f"❌ LLM initialization failed: {e}")
            self.llm = None
            self.chat_llm = None

    def _initialize_memory(self):
        """Initialize conversation memory"""
        if not self.llm:
            return
        
        try:
            if self.memory_type == "summary":
                self.memory = ConversationSummaryBufferMemory(
                    llm=self.llm,
                    max_token_limit=self.max_memory_tokens,
                    return_messages=True,
                    memory_key="chat_history"
                )
            else:
                self.memory = ConversationBufferMemory(
                    return_messages=True,
                    memory_key="chat_history"
                )
            
            print(f"✅ Memory initialized: {self.memory_type}")
        except Exception as e:
            print(f"❌ Memory initialization failed: {e}")
            self.memory = None

    def _initialize_tools(self):
        """Initialize external tools"""
        self.tools = []
        
        # Calculator tool
        self.tools.append(Tool(
            name="Calculator",
            func=self._calculate,
            description="Useful for mathematical calculations. Input should be a mathematical expression."
        ))
        
        # Time/Date tool
        self.tools.append(Tool(
            name="DateTime",
            func=self._get_datetime_info,
            description="Get current date, time, or day of week information."
        ))
        
        # Search tool (DuckDuckGo - no API key required)
        try:
            search = DuckDuckGoSearchRun()
            self.tools.append(Tool(
                name="WebSearch",
                func=search.run,
                description="Search the web for current information. Use when you need recent data or facts."
            ))
            print("✅ Web search tool initialized")
        except Exception as e:
            print(f"⚠️ Web search tool failed: {e}")
        
        # Wikipedia tool
        try:
            wikipedia = WikipediaAPIWrapper()
            self.tools.append(Tool(
                name="Wikipedia",
                func=wikipedia.run,
                description="Search Wikipedia for encyclopedic information about people, places, events, etc."
            ))
            print("✅ Wikipedia tool initialized")
        except Exception as e:
            print(f"⚠️ Wikipedia tool failed: {e}")
        
        # Weather tool (placeholder - you can integrate with a weather API)
        self.tools.append(Tool(
            name="Weather",
            func=self._get_weather,
            description="Get weather information for a location. (Currently placeholder)"
        ))
        
        print(f"✅ {len(self.tools)} tools initialized")

    def _initialize_agent(self):
        """Initialize the agent with tools"""
        if not self.llm or not self.tools:
            return
        
        try:
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.chat_llm,  # Use chat model for better tool use
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                memory=self.memory,
                verbose=False,
                handle_parsing_errors=True,
                max_iterations=3,
                early_stopping_method="generate"
            )
            print("✅ Agent initialized with tools")
        except Exception as e:
            print(f"❌ Agent initialization failed: {e}")
            self.agent = None

    def _calculate(self, expression: str) -> str:
        """Calculator tool implementation"""
        try:
            # Basic safety check for allowed characters
            allowed_chars = set('0123456789+-*/().% ')
            if not all(c in allowed_chars for c in expression.replace(' ', '')):
                return "Error: Invalid characters in expression"
            
            # Evaluate the expression
            result = eval(expression)
            return f"The result of {expression} is {result}"
        except ZeroDivisionError:
            return "Error: Division by zero"
        except Exception as e:
            return f"Error: Unable to calculate '{expression}' - {str(e)}"

    def _get_datetime_info(self, query: str = "") -> str:
        """Get current date/time information"""
        now = datetime.now()
        query_lower = query.lower()
        
        if "date" in query_lower:
            return f"Today's date is {now.strftime('%A, %B %d, %Y')}"
        elif "time" in query_lower:
            return f"Current time is {now.strftime('%I:%M:%S %p')}"
        elif "day" in query_lower:
            return f"Today is {now.strftime('%A')}"
        else:
            return f"Current date and time: {now.strftime('%A, %B %d, %Y at %I:%M:%S %p')}"

    def _get_weather(self, location: str = "") -> str:
        """Weather tool placeholder"""
        return f"Weather information for {location or 'your location'} is not available yet. Please check your local weather service or integrate with a weather API."

    def ask(self, question: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Ask a question and get a response
        
        Args:
            question: The question to ask
            context: Optional context for the question
            
        Returns:
            Dictionary with response, metadata, and conversation info
        """
        if not question.strip():
            return {
                "response": "Please ask me a question.",
                "error": "Empty question",
                "timestamp": datetime.now().isoformat(),
                "model_used": self.model_name,
                "tokens_used": 0
            }
        
        start_time = time.time()
        
        try:
            # Prepare the question with context if provided
            if context:
                formatted_question = f"Context: {context}\n\nQuestion: {question}"
            else:
                formatted_question = question
            
            # Get response based on available components
            if self.agent:
                # Use agent with tools
                response = self.agent.run(formatted_question)
            elif self.chat_llm and self.memory:
                # Use chat model with memory
                conversation = ConversationChain(
                    llm=self.chat_llm,
                    memory=self.memory,
                    verbose=False
                )
                response = conversation.predict(input=formatted_question)
            elif self.chat_llm:
                # Use chat model without memory
                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=formatted_question)
                ]
                result = self.chat_llm.invoke(messages)
                response = result.content
            elif self.llm:
                # Fallback to basic LLM
                response = self.llm.invoke(formatted_question).content
            else:
                response = "I'm sorry, but I'm currently unable to process your question due to configuration issues."
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Store in conversation history
            self.conversation_history.append({
                "question": question,
                "response": response,
                "timestamp": datetime.now().isoformat(),
                "response_time": response_time
            })
            
            return {
                "response": response,
                "question": question,
                "timestamp": datetime.now().isoformat(),
                "response_time": response_time,
                "model_used": self.model_name,
                "tools_available": len(self.tools),
                "memory_enabled": self.memory is not None,
                "conversation_length": len(self.conversation_history)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing question: {e}")
            return {
                "response": f"I encountered an error while processing your question: {str(e)}",
                "error": str(e),
                "question": question,
                "timestamp": datetime.now().isoformat(),
                "model_used": self.model_name
            }

    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Get conversation history"""
        if limit:
            return self.conversation_history[-limit:]
        return self.conversation_history

    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        if self.memory:
            self.memory.clear()
        print("🧹 Conversation history cleared")

    def save_conversation(self, filename: str):
        """Save conversation history to file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
            print(f"💾 Conversation saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save conversation: {e}")

    def load_conversation(self, filename: str):
        """Load conversation history from file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.conversation_history = json.load(f)
            print(f"📂 Conversation loaded from {filename}")
        except Exception as e:
            print(f"❌ Failed to load conversation: {e}")

    def get_memory_summary(self) -> str:
        """Get a summary of the conversation memory"""
        if not self.memory:
            return "Memory not enabled"
        
        try:
            if hasattr(self.memory, 'buffer'):
                return f"Memory contains {len(self.memory.buffer)} messages"
            else:
                return "Memory is active but details unavailable"
        except Exception as e:
            return f"Memory status unknown: {e}"

    def set_system_prompt(self, prompt: str):
        """Set a custom system prompt for the AI"""
        self.system_prompt = prompt
        print(f"🎯 System prompt updated")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "memory_enabled": self.memory is not None,
            "memory_type": self.memory_type,
            "tools_count": len(self.tools),
            "agent_enabled": self.agent is not None,
            "conversation_count": len(self.conversation_history)
        }

    def test_connection(self) -> bool:
        """Test the connection to OpenRouter API"""
        try:
            test_response = self.ask("Hello, this is a connection test.")
            return "error" not in test_response
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False

    def benchmark_response_time(self, num_tests: int = 5) -> Dict[str, float]:
        """Benchmark response times"""
        test_questions = [
            "What is 2+2?",
            "What day is today?",
            "Tell me a joke",
            "What is the capital of France?",
            "How are you doing?"
        ]
        
        times = []
        for i in range(min(num_tests, len(test_questions))):
            start = time.time()
            self.ask(test_questions[i])
            times.append(time.time() - start)
        
        return {
            "average_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "total_tests": len(times)
        }

    def interactive_mode(self):
        """Start interactive Q&A mode"""
        print("🤖 EVA QA Engine - Interactive Mode")
        print("=" * 40)
        print("Type 'quit', 'exit', or 'bye' to stop")
        print("Type 'help' for available commands")
        print("Type 'clear' to clear conversation history")
        print("Type 'save <filename>' to save conversation")
        print("Type 'info' to see model information")
        print()
        
        while True:
            try:
                question = input("❓ Your question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                if question.lower() == 'help':
                    print("""
Available commands:
- quit/exit/bye: Exit the program
- clear: Clear conversation history
- save <filename>: Save conversation to file
- load <filename>: Load conversation from file
- info: Show model information
- memory: Show memory summary
- benchmark: Test response times
- test: Test API connection
""")
                    continue
                
                if question.lower() == 'clear':
                    self.clear_conversation_history()
                    continue
                
                if question.lower().startswith('save '):
                    filename = question[5:].strip() or f"conversation_{int(time.time())}.json"
                    self.save_conversation(filename)
                    continue
                
                if question.lower().startswith('load '):
                    filename = question[5:].strip()
                    self.load_conversation(filename)
                    continue
                
                if question.lower() == 'info':
                    info = self.get_model_info()
                    for key, value in info.items():
                        print(f"  {key}: {value}")
                    continue
                
                if question.lower() == 'memory':
                    print(f"  {self.get_memory_summary()}")
                    continue
                
                if question.lower() == 'benchmark':
                    print("🏃 Running benchmark...")
                    results = self.benchmark_response_time()
                    for key, value in results.items():
                        print(f"  {key}: {value:.2f}s" if 'time' in key else f"  {key}: {value}")
                    continue
                
                if question.lower() == 'test':
                    print("🧪 Testing connection...")
                    success = self.test_connection()
                    print("✅ Connection successful" if success else "❌ Connection failed")
                    continue
                
                # Process the question
                result = self.ask(question)
                print(f"🤖 EVA: {result['response']}")
                print(f"  (Response time: {result.get('response_time', 0):.2f}s)")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

class SimpleQAEngine:
    """Simplified QA Engine for basic use cases"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.conversation = []

    def ask(self, question: str) -> str:
        """Simple question asking without advanced features"""
        if not self.api_key:
            return "OpenRouter API key not configured"
        
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://your-site.com',
                'X-Title': 'EVA Simple QA'
            }
            
            self.conversation.append({"role": "user", "content": question})
            
            data = {
                'model': 'anthropic/claude-3.5-sonnet',
                'messages': [
                    {"role": "system", "content": "You are EVA, a helpful and emotionally intelligent AI assistant."},
                    *self.conversation[-10:]  # Keep last 10 messages
                ],
                'max_tokens': 500,
                'temperature': 0.7
            }
            
            response = requests.post(
                'https://openrouter.ai/api/v1/chat/completions',
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                answer = response.json()['choices'][0]['message']['content']
                self.conversation.append({"role": "assistant", "content": answer})
                return answer
            else:
                return f"OpenRouter API Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error: {str(e)}"

# Factory function
def create_qa_engine(simple: bool = False, **kwargs) -> Union[EnhancedQAEngine, SimpleQAEngine]:
    """
    Factory function to create QA engine
    
    Args:
        simple: If True, creates SimpleQAEngine, otherwise EnhancedQAEngine
        **kwargs: Arguments passed to EnhancedQAEngine
        
    Returns:
        QA Engine instance
    """
    if simple:
        return SimpleQAEngine(kwargs.get('api_key'))
    else:
        return EnhancedQAEngine(**kwargs)

# Convenience functions
def quick_ask(question: str, api_key: Optional[str] = None) -> str:
    """Quick question without maintaining state"""
    engine = SimpleQAEngine(api_key)
    return engine.ask(question)

if __name__ == "__main__":
    # Test the QA engine
    print("🧪 Testing QA Engine...")
    
    # Test enhanced engine
    qa_engine = EnhancedQAEngine()
    
    if qa_engine.llm:
        print("\n🤖 Enhanced QA Engine Test:")
        
        # Test basic functionality
        result = qa_engine.ask("What is 2+2?")
        print(f"Q: What is 2+2?")
        print(f"A: {result['response']}")
        
        # Test with context
        result = qa_engine.ask("What's the weather like?", context="User is in New York")
        print(f"\nQ: What's the weather like? (Context: User is in New York)")
        print(f"A: {result['response']}")
        
        # Show model info
        print(f"\n📊 Model Info:")
        info = qa_engine.get_model_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Start interactive mode if run directly
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
            qa_engine.interactive_mode()
    else:
        print("⚠️ Enhanced QA Engine not available, testing simple engine...")
        simple_engine = SimpleQAEngine()
        response = simple_engine.ask("Hello, how are you?")
        print(f"Simple Engine Response: {response}")
