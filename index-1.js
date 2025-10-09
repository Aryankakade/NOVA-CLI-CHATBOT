// BACKEND CONFIGURATION - EXACT MATCH WITH backend.py
const API_BASE = "http://127.0.0.1:8000"; // Change to your backend URL

// Global Variables
let currentUser = null;
let currentPage = 1;
let selectedAgent = null;
let isRecording = false;
let mediaRecorder;
let audioChunks = [];
let chatHistory = [];
let currentSession = null;
let isProcessing = false;
let userId = null;
let voiceModeEnabled = false;
const VOICE_PROCESS_ENDPOINT = `${API_BASE}/voice/process`;


// COMPLETE FORM SUBMISSION PREVENTION
document.addEventListener('DOMContentLoaded', function () {
    // Prevent ALL form submissions globally
    document.addEventListener('submit', function (e) {
        e.preventDefault();
    });

    // Handle Enter key inside chatInput
    const chatInput = document.getElementById('chatInput');
    chatInput.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // stop page reload
            e.stopPropagation();
            sendMessage();
            return false;
        }
    });

    // Initialize app
    initializeApp();
});

function initializeApp() {
    console.log('🚀 NOVA Ultra Professional Frontend Initializing...');
    console.log('🔗 Backend API:', API_BASE);
    
    // Initialize user ID
    userId = localStorage.getItem('nova_user_id') || `web-user-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    localStorage.setItem('nova_user_id', userId);
    
    createStars();
    setupEventListeners();
    checkLoginState();
    initializeSystemCheck();
}

// ========== BACKEND API INTEGRATION - ALL ENDPOINTS ==========

// EXACT BACKEND INTEGRATION: GET / endpoint
async function getSystemInfo() {
    try {
        const response = await fetch(`${API_BASE}/`, {
            method: 'GET'
        });
        
        if (response.ok) {
            const data = await response.json();
            console.log('✅ System Info:', data);
            return data;
        } else {
            throw new Error(`System info failed: ${response.status}`);
        }
    } catch (error) {
        console.error('❌ System info error:', error);
        return null;
    }
}

// EXACT BACKEND INTEGRATION: POST /chat endpoint with ChatRequest/ChatResponse models
async function sendMessage() {
    if (isProcessing) return;
    
    const input = document.getElementById('chatInput');
    if (!input) return;
    
    const message = input.value.trim();
    if (!message) return;

    isProcessing = true;

    // 🔊 User pressed Enter / Send button → click beep
    soundManager.playBeep("click");
    
    // Clear input immediately
    input.value = '';
    autoResize();
    
    // Check for special commands first
    if (await handleSpecialCommands(message)) {
        isProcessing = false;
        return;
    }
    
    addMessageToChat('user', message);
    const typingId = addTypingIndicator();

    try {
        // EXACT backend POST /chat endpoint with ChatRequest model
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                user_id: userId,
                agent_type: selectedAgent
            })
        });

        if (!response.ok) {
            throw new Error(`Backend error: ${response.status} - ${response.statusText}`);
        }

        const data = await response.json();
        removeTypingIndicator(typingId);
        
        // Process ChatResponse model response
        const botResponse = data.response || 'No response received.';
        const agentUsed = data.agent_used || selectedAgent || 'general';
        const responseTime = data.response_time || 0;
        const language = data.language || 'english';
        const emotion = data.emotion || 'neutral';
        const emotionConfidence = data.emotion_confidence || 0.0;
        const agentConfidence = data.agent_confidence || 0.0;
        const conversationCount = data.conversation_count || 0;
        const fileContextUsed = data.file_context_used || false;
        const sessionId = data.session_id || 'unknown';
        
        addMessageToChat('bot', botResponse, { 
            agent: agentUsed, 
            responseTime: responseTime,
            language: language,
            emotion: emotion,
            emotionConfidence: emotionConfidence,
            agentConfidence: agentConfidence,
            conversationCount: conversationCount,
            fileContextUsed: fileContextUsed,
            sessionId: sessionId
        });

        // 🔊 AI responded → success beep
        soundManager.playBeep("success");
        
        saveChatToHistory(message, botResponse);
        
        // Update selected agent if backend switched agents
        if (agentUsed !== selectedAgent) {
            selectedAgent = agentUsed;
            const agentInfo = getAgentInfo(agentUsed);
            const chatAgentNameEl = document.getElementById('chatAgentName');
            const chatAgentDescEl = document.getElementById('chatAgentDesc');
            
            if (chatAgentNameEl) chatAgentNameEl.textContent = agentInfo.name;
            if (chatAgentDescEl) chatAgentDescEl.textContent = agentInfo.description;
            
            showNotification(`Switched to ${agentInfo.name} 🔄`, 'info');
        }

        console.log('✅ Chat Response:', {
            agent: agentUsed,
            language: language,
            emotion: emotion,
            responseTime: responseTime,
            conversationCount: conversationCount,
            fileContextUsed: fileContextUsed
        });

    } catch (err) {
        console.error('❌ Chat error:', err);
        removeTypingIndicator(typingId);
        addMessageToChat('bot', `❌ Connection Error: ${err.message}. Please ensure the NOVA backend is running on ${API_BASE}`);

        // 🔊 Error beep
        soundManager.playBeep("error");
    }
    
    isProcessing = false;
}

// EXACT BACKEND INTEGRATION: POST /file/upload endpoint
async function processFileUpload() {
    const fileInput = document.getElementById('fileInput');
    const filePromptEl = document.getElementById('filePrompt');
    
    if (!fileInput || !filePromptEl) return;
    
    const file = fileInput.files[0];
    const prompt = filePromptEl.value.trim();
    
    if (!file) {
        showNotification('Please select a file first', 'error');
        soundManager.playBeep("error");
        return;
    }
    
    closeFileModal();
    
    const displayMessage = prompt || `Analyze this file: ${file.name}`;
    addMessageToChat('user', `📎 [File: ${file.name}] ${displayMessage}`);
    const typingId = addTypingIndicator();
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('user_id', userId);
        if (prompt) formData.append('prompt', prompt);

        const response = await fetch(`${API_BASE}/file/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`File upload error: ${response.status}`);
        }
        
        const data = await response.json();
        removeTypingIndicator(typingId);

        if (data.success && data.metadata?.file_analysis) {
            const analysis = data.metadata.file_analysis;

            let responseText = `📄 **File Uploaded!**\n\n`;
            responseText += `**File:** ${analysis.file_name}\n`;
            responseText += `**Type:** ${analysis.file_type}\n`;
            responseText += `**Size:** ${formatFileSize(analysis.file_size)}\n`;
            if (analysis.lines) responseText += `**Lines:** ${analysis.lines}\n`;
            if (analysis.words) responseText += `**Words:** ${analysis.words}\n`;
            if (analysis.chars) responseText += `**Characters:** ${analysis.chars}\n`;

            // ✅ Show AI’s answer (from standardized field)
            responseText += `\n🤖 **AI Response:**\n${data.response || 'No AI response generated.'}`;
            
            addMessageToChat('bot', responseText);
            soundManager.playBeep("success");

        } else {
            throw new Error(data.error || data.message || 'File processing failed');
        }
        
        saveChatToHistory(
            `[File: ${file.name}] ${displayMessage}`,
            data.response || 'File processed successfully'
        );
        showNotification('File processed successfully! 📄', 'success');
        
    } catch (error) {
        console.error('❌ File upload error:', error);
        removeTypingIndicator(typingId);
        addMessageToChat('bot', `❌ File processing failed: ${error.message}`);
        soundManager.playBeep("error");
        showNotification('File processing failed: ' + error.message, 'error');
    }
}

// EXACT BACKEND INTEGRATION: POST /voice/speak endpoint
async function speakText(text) {
    if (!text) return false;
    
    try {
        showNotification('🔊 Generating speech...', 'info');
        
        const formData = new FormData();
        formData.append('text', text);
        formData.append('user_id', userId);

        const response = await fetch(VOICE_PROCESS_ENDPOINT, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`TTS error: ${response.status}`);
        }
        
        // Get audio blob and play it
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        
        audio.onloadeddata = () => {
            showNotification('🔊 Playing audio...', 'success');
            soundManager.playBeep("success");   // ✅ double beep when audio is ready
        };
        
        audio.onended = () => {
            URL.revokeObjectURL(audioUrl);
            showNotification('🔊 Audio playback complete', 'success');
            soundManager.playBeep("success");   // ✅ beep after playback ends
        };
        
        audio.onerror = () => {
            showNotification('❌ Audio playback failed', 'error');
            soundManager.playBeep("error");     // ❌ error beep
        };
        
        await audio.play();
        return true;
        
    } catch (error) {
        console.error('TTS error:', error);
        showNotification('TTS error: ' + error.message, 'error');
        soundManager.playBeep("error");         // ❌ error beep
        return false;
    }
}

async function startVoiceRecording() {
    try {
        // Request microphone access
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
        audioChunks = [];
        
        // Setup recording handlers
        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) {
                audioChunks.push(e.data);
            }
        };
        
        mediaRecorder.onstop = async () => {
            await processAudioRecording();
            stream.getTracks().forEach(track => track.stop());
        };
        
        // Start recording
        mediaRecorder.start(100); // Collect data every 100ms
        isRecording = true;
        updateVoiceButtonState();
        showNotification('Recording... Speak now', 'info');
        soundManager.playBeep("click");          // 🎙️ click beep when recording starts
        
        // Auto-stop after 10 seconds
        setTimeout(() => {
            if (isRecording) stopVoiceRecording();
        }, 10000);
        
    } catch (err) {
        console.error('Microphone error:', err);
        showNotification('Could not access microphone', 'error');
        soundManager.playBeep("error");         // ❌ error beep
    }
}

function stopVoiceRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        updateVoiceButtonState();
        soundManager.playBeep("click");         // 🔊 beep on stop
    }
}

async function processAudioRecording() {
    showLoading('Processing your voice...');
    
    try {
        // Combine audio chunks into a single blob
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        const formData = new FormData();
        formData.append('audio', audioBlob, 'voice-input.webm');
        formData.append('user_id', userId);
        
        // Send to backend
        const response = await fetch(`${API_BASE}/voice/process`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}`);
        }
        
        // Play the response
        const audioData = await response.blob();
        const audioUrl = URL.createObjectURL(audioData);
        const audioPlayer = new Audio(audioUrl);

        audioPlayer.onerror = (e) => {
            console.error('Audio playback error:', e);
            showNotification('Failed to play audio response', 'error');
            soundManager.playBeep("error");     // ❌ error beep
        };
        
        audioPlayer.onended = () => {
            URL.revokeObjectURL(audioUrl);
            showNotification('Voice response complete', 'success');
            soundManager.playBeep("success");   // ✅ success beep after voice reply
        };
        
        await audioPlayer.play();
        soundManager.playBeep("success");       // ✅ beep when AI voice starts playing
        
    } catch (error) {
        console.error('Voice processing failed:', error);
        showNotification('Voice processing failed', 'error');
        soundManager.playBeep("error");         // ❌ error beep
    } finally {
        hideLoading();
    }
}

function updateVoiceButtonState() {
    const voiceBtn = document.getElementById('voiceBtn');
    if (voiceBtn) {
        voiceBtn.innerHTML = isRecording 
            ? '<i class="fas fa-microphone-slash"></i>' 
            : '<i class="fas fa-microphone"></i>';
        voiceBtn.classList.toggle('recording', isRecording);
    }
}

// EXACT BACKEND INTEGRATION: POST /web/search endpoint with SearchRequest model
async function performWebSearch(query) {
    if (!query.trim()) return;
    
    addMessageToChat('user', `🔍 Search: ${query}`);
    const typingId = addTypingIndicator();
    soundManager.playBeep("click");   // ✅ beep when user starts search
    
    try {
        // Call backend /web/search
        const response = await fetch(`${API_BASE}/web/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                user_id: userId
            })
        });
        
        if (!response.ok) {
            throw new Error(`Web search error: ${response.status}`);
        }
        
        const data = await response.json();
        removeTypingIndicator(typingId);
        
        if (data.success) {
            let formattedResponse = "🌐 **Search Results:**\n\n";
            
            if (data.results && data.results.length > 0) {
                data.results.forEach((r, i) => {
                    formattedResponse += `**${i + 1}. ${r.title}**\n${r.snippet || ''}\n🔗 ${r.url}\n\n`;
                });
            } else {
                formattedResponse += "No results found.\n";
            }

            // ✅ Add AI Smart Summary if available
            if (data.summary_answer) {
                formattedResponse = `🤖 **Smart Answer:**\n${data.summary_answer}\n\n---\n` + formattedResponse;
            }
            
            addMessageToChat('bot', formattedResponse);
            showNotification('Web search completed! 🔍', 'success');
            soundManager.playBeep("success");   // ✅ double beep on success
            
            console.log('✅ Web Search Results:', data);
        } else {
            throw new Error(data.error || 'Web search failed');
        }
        
    } catch (error) {
        removeTypingIndicator(typingId);
        console.error('❌ Web search error:', error);
        addMessageToChat('bot', `❌ Web search failed: ${error.message}`);
        showNotification('Web search failed: ' + error.message, 'error');
        soundManager.playBeep("error");   // ❌ error beep
    }
}

// EXACT BACKEND INTEGRATION: POST /github/analyze endpoint with GitHubRequest model
// 🔍 Analyze GitHub Repo
async function analyzeGitHubRepo() {
    const repoUrlInput = document.getElementById('githubRepoUrl');
    if (!repoUrlInput) return;

    const repoUrl = repoUrlInput.value.trim();
    if (!repoUrl) {
        showNotification('Please enter a GitHub repository URL', 'error');
        soundManager.playBeep("error");
        return;
    }
    if (!isValidGitHubUrl(repoUrl)) {
        showNotification('Please enter a valid GitHub repository URL', 'error');
        soundManager.playBeep("error");
        return;
    }

    showLoading('🔍 Analyzing GitHub repository...');
    soundManager.playBeep("click");

    try {
        const formData = new FormData();
        formData.append('repo_url', repoUrl);
        formData.append('user_id', userId || 'web-user');

        const response = await fetch(`${API_BASE}/github/analyze`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error(`GitHub analysis error: ${response.status}`);
        const data = await response.json();

        hideLoading();

        if (data.success) {
            const repoAnalysisEl = document.getElementById('repoAnalysis');
            const codeQualityEl = document.getElementById('codeQuality');
            const debugSuggestionsEl = document.getElementById('debugSuggestions');

            const repoUrlMeta = data.metadata?.repo_url || repoUrl;
            const rawInsights = data.metadata?.raw_insights || {};

            if (repoAnalysisEl) {
                repoAnalysisEl.textContent =
`📂 Repository Analysis
Repository: ${repoUrlMeta}

--- Detailed Analysis ---
${rawInsights.code_quality || 'No code quality review available'}

${rawInsights.debugging || 'No debugging suggestions available'}

🤖 AI Response:
${data.response || 'No AI response generated'}
`;
            }

            if (codeQualityEl) {
                codeQualityEl.textContent = rawInsights.code_quality || 'No critical issues detected';
            }
            if (debugSuggestionsEl) {
                debugSuggestionsEl.textContent = rawInsights.debugging || 'No specific suggestions available';
            }

            showNotification('Repository analyzed successfully! 🎉', 'success');
            soundManager.playBeep("success");
            console.log('✅ GitHub Analysis Complete:', data);

        } else {
            throw new Error(data.error || 'GitHub analysis failed');
        }

    } catch (error) {
        hideLoading();
        console.error('❌ GitHub analysis error:', error);
        showNotification('GitHub analysis failed: ' + error.message, 'error');
        soundManager.playBeep("error");

        const repoAnalysisEl = document.getElementById('repoAnalysis');
        const codeQualityEl = document.getElementById('codeQuality');
        const debugSuggestionsEl = document.getElementById('debugSuggestions');

        if (repoAnalysisEl) repoAnalysisEl.textContent = 'Analysis failed: ' + error.message;
        if (codeQualityEl) codeQualityEl.textContent = 'Unable to assess code quality';
        if (debugSuggestionsEl) debugSuggestionsEl.textContent = 'Unable to provide suggestions';
    }
}

// 💬 Ask Follow-up Question on Repo
async function askGitHubQuestion(question) {
    if (!question.trim()) return;

    soundManager.playBeep("click");

    try {
        const formData = new FormData();
        formData.append('question', question);

        const response = await fetch(`${API_BASE}/github/question`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error(`GitHub question error: ${response.status}`);
        const data = await response.json();

        if (data.success) {
            soundManager.playBeep("success");
            return data.response;   // ✅ standardized field
        } else {
            throw new Error(data.error || 'GitHub question failed');
        }

    } catch (error) {
        console.error('❌ GitHub question error:', error);
        soundManager.playBeep("error");
        return `Error asking GitHub question: ${error.message}`;
    }
}

// EXACT BACKEND INTEGRATION: GET /system/status endpoint
async function getSystemStatus() {
    try {
        const response = await fetch(`${API_BASE}/system/status`, {
            method: 'GET'
        });
        
        if (response.ok) {
            const status = await response.json();
            console.log('✅ System Status:', status);
            return status;
        } else {
            throw new Error(`Status check failed: ${response.status}`);
        }
    } catch (error) {
        console.error('❌ System status error:', error);
        return null;
    }
}

// EXACT BACKEND INTEGRATION: POST /clear/{user_id} endpoint
async function clearContext() {
    try {
        const response = await fetch(`${API_BASE}/clear/${userId}`, {
            method: 'POST'
        });
        
        if (response.ok) {
            const data = await response.json();
            if (data.success) {
                showNotification('Context cleared successfully! 🧹', 'success');
                
                // Clear chat interface
                const messagesContainer = document.getElementById('chatMessages');
                if (messagesContainer) {
                    messagesContainer.innerHTML = '';
                    const agentInfo = getAgentInfo(selectedAgent || 'general');
                    addMessageToChat('bot', `Hello! I'm ${agentInfo.name}. My context has been cleared. How can I help you with a fresh start?`);
                }
                
                localStorage.removeItem('chatHistory');
                chatHistory = [];
                currentSession = generateSessionId();
                
                console.log('✅ Context cleared successfully');
                return true;
            }
        } else {
            throw new Error(`Clear context failed: ${response.status}`);
        }
    } catch (error) {
        console.error('❌ Clear context error:', error);
        showNotification('Failed to clear context: ' + error.message, 'error');
        return false;
    }
}

// EXACT BACKEND INTEGRATION: GET /health endpoint
async function healthCheck() {
    try {
        const response = await fetch(`${API_BASE}/health`, {
            method: 'GET'
        });
        
        if (response.ok) {
            const health = await response.json();
            console.log('✅ Backend Health Check:', health);
            return health;
        } else {
            throw new Error(`Health check failed: ${response.status}`);
        }
    } catch (error) {
        console.error('❌ Backend health check failed:', error);
        return null;
    }
}

// ========== SPECIAL COMMANDS HANDLING ==========
async function handleSpecialCommands(message) {
    const command = message.toLowerCase().trim();
    
    if (command.startsWith('/search ') || command.startsWith('/web ')) {
        const query = message.substring(command.indexOf(' ') + 1);
        await performWebSearch(query);
        return true;
    }
    
    if (command === '/clear' || command === '/reset') {
        await clearContext();
        return true;
    }
    
    if (command === '/status' || command === '/health') {
        const status = await getSystemStatus();
        if (status) {
            let statusText = '📊 **System Status:**\n\n';
            
            statusText += '**Core Systems:**\n';
            for (const [key, value] of Object.entries(status.core_systems || {})) {
                statusText += `• ${key}: ${value}\n`;
            }
            
            statusText += '\n**Premium Systems:**\n';
            for (const [key, value] of Object.entries(status.premium_systems || {})) {
                statusText += `• ${key}: ${value}\n`;
            }
            
            statusText += '\n**Agents:**\n';
            for (const [key, value] of Object.entries(status.agents || {})) {
                statusText += `• ${key}: ${value}\n`;
            }
            
            statusText += '\n**Session Info:**\n';
            for (const [key, value] of Object.entries(status.session_info || {})) {
                statusText += `• ${key}: ${value}\n`;
            }
            
            addMessageToChat('bot', statusText);
        } else {
            addMessageToChat('bot', '❌ Unable to retrieve system status');
        }
        return true;
    }
    
    if (command === '/help' || command === '/commands') {
        const helpText = `🤖 **Available Commands:**

🔍 **Chat Commands:**
• Just type normally to chat with NOVA
• Use the voice button 🎤 for speech playback
• Use the paperclip 📎 to upload files

🔍 **Search Commands:**
• \`/search [query]\` - Search the web
• \`/web [query]\` - Alternative web search

🛠️ **System Commands:**
• \`/clear\` - Clear conversation context
• \`/reset\` - Same as clear
• \`/status\` - Show system status
• \`/health\` - Show backend health
• \`/help\` - Show this help message

🎯 **Special Features:**
• File upload and analysis
• GitHub repository analysis
• Multi-agent AI system
• Voice output (TTS)
• Memory persistence

Just start typing to begin chatting with NOVA! 🚀`;
        
        addMessageToChat('bot', helpText);
        return true;
    }
    
    if (command.startsWith('/github ')) {
        const repoUrl = message.substring(8).trim();
        if (isValidGitHubUrl(repoUrl)) {
            addMessageToChat('user', `🔍 Analyzing GitHub repository: ${repoUrl}`);
            // Set the repo URL and trigger analysis
            const repoUrlInput = document.getElementById('githubRepoUrl');
            if (repoUrlInput) {
                repoUrlInput.value = repoUrl;
                analyzeGitHubRepo();
            }
        } else {
            addMessageToChat('bot', '❌ Invalid GitHub URL. Please provide a valid repository URL.');
        }
        return true;
    }
    
    if (command.startsWith('/ask ') && command.includes('github')) {
        const question = message.substring(5).trim();
        const answer = await askGitHubQuestion(question);
        addMessageToChat('bot', `**GitHub Repository Answer:**\n\n${answer}`);
        return true;
    }
    
    return false;
}

// ========== AUTHENTICATION SYSTEM (Simple Demo Mode) ==========
function checkLoginState() {
    const isLoggedIn = localStorage.getItem('isLoggedIn');
    const profileData = localStorage.getItem('profileData');
    
    if (isLoggedIn === 'true') {
        if (profileData) {
            navigateToPage(4);
            initializeChatInterface();
        } else {
            navigateToPage(2);
        }
    } else {
        navigateToPage(1);
        addSimpleLoginForm();
        showLoginHints();
    }
}

function addSimpleLoginForm() {
    const page1 = document.getElementById('page1');
    if (page1 && !document.getElementById('simpleLoginForm')) {
        const formDiv = document.createElement('div');
        formDiv.id = 'simpleLoginForm';
        formDiv.className = 'mt-8';
        formDiv.innerHTML = `
            <div class="space-y-4 max-w-md mx-auto">
                <input type="email" id="loginEmail" placeholder="Email" value="user@gmail.com"
                       class="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:border-cyan-500 focus:outline-none">
                <input type="password" id="loginPassword" placeholder="Password" value="password123"
                       class="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:border-cyan-500 focus:outline-none">
                <button type="button" id="simpleLoginBtn"
                        class="w-full bg-gradient-to-r from-cyan-500 to-blue-600 text-white py-3 px-4 rounded-lg font-semibold hover:from-cyan-600 hover:to-blue-700 transition-all duration-200 transform hover:scale-105">
                    🚀 Sign In to NOVA
                </button>
            </div>
        `;
        page1.insertBefore(formDiv, page1.firstChild);
    }
}

function showLoginHints() {
    const loginContainer = document.querySelector('.login-container') || document.getElementById('page1');
    if (loginContainer && !document.getElementById('loginHints')) {
        const hintsDiv = document.createElement('div');
        hintsDiv.id = 'loginHints';
        hintsDiv.className = 'text-center mt-6 p-4 bg-gray-800 bg-opacity-50 rounded-lg border border-gray-700';
        hintsDiv.innerHTML = `
            <p class="text-gray-400 text-sm mb-2">Demo Credentials:</p>
            <p class="text-gray-500 text-sm">📧 Email: <span class="text-gray-400">user@gmail.com</span></p>
            <p class="text-gray-500 text-sm">🔑 Password: <span class="text-gray-400">password123</span></p>
        `;
        loginContainer.appendChild(hintsDiv);
    }
}

function handleSimpleLogin() {
    if (isProcessing) return;
    isProcessing = true;

    const emailInput = document.getElementById('loginEmail');
    const passwordInput = document.getElementById('loginPassword');
    const email = emailInput ? emailInput.value.trim() : 'user@gmail.com';
    const password = passwordInput ? passwordInput.value.trim() : 'password123';

    if (email === 'user@gmail.com' && password === 'password123') {
        showLoading('🚀 Signing you into NOVA...');
        localStorage.setItem('isLoggedIn', 'true');
        localStorage.setItem('userEmail', email);
        currentUser = { email: email, displayName: 'NOVA User' };
        
        setTimeout(() => {
            hideLoading();
            const profileData = localStorage.getItem('profileData');
            if (profileData) {
                navigateToPage(4);
                initializeChatInterface();
                showNotification('Welcome back to NOVA! 🎉', 'success');
            } else {
                navigateToPage(2);
                showNotification('Let\'s set up your profile! ⚙️', 'info');
            }
            isProcessing = false;
        }, 1500);
    } else {
        showNotification('Invalid credentials! Use: user@gmail.com / password123', 'error');
        isProcessing = false;
    }
}

function handleLogout() {
    localStorage.removeItem('isLoggedIn');
    localStorage.removeItem('userEmail');
    localStorage.removeItem('profileData');
    localStorage.removeItem('chatHistory');
    currentUser = null;
    selectedAgent = null;
    currentSession = null;
    navigateToPage(1);
    showNotification('Logged out successfully! See you soon! 👋', 'success');
}

function startNewChat() {
    const chatMessages = document.getElementById('chatMessages');
    const chatInput = document.getElementById('chatInput');

    // Reset chat messages
    chatMessages.innerHTML = `
        <div class="message bot">
            <div class="font-medium text-cyan-400 mb-1">NOVA</div>
            <div>Hello! I'm NOVA, your AI assistant. Let's start a new conversation. 🚀</div>
        </div>
    `;

    // Reset input
    chatInput.value = '';

    // Close floating menu
    document.getElementById('floatingMenu').classList.remove('active');
}

class SoundManager {
    constructor() {
        this.enabled = true;
    }

    playBeep(type) {
        if (!this.enabled) return;

        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = ctx.createOscillator();
        const gainNode = ctx.createGain();

        oscillator.connect(gainNode);
        gainNode.connect(ctx.destination);

        let duration = 0.1;
        let frequency = 800;

        if (type === "click") {
            frequency = 800;
            duration = 0.1;
        } else if (type === "success") {
            frequency = 600;
            duration = 0.2;
        } else if (type === "error") {
            frequency = 400;
            duration = 0.3;
        } else if (type === "notification") {
            frequency = 1000;
            duration = 0.15;
        }

        oscillator.type = "sine";
        oscillator.frequency.setValueAtTime(frequency, ctx.currentTime);
        oscillator.start();
        oscillator.stop(ctx.currentTime + duration);
    }
}

// global instance
const soundManager = new SoundManager();

// ========== PROFILE SETUP ==========
function handleProfileSubmit(e) {
    if (e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    if (isProcessing) return false;
    isProcessing = true;
    
    const nameInput = document.getElementById('userName');
    const ageInput = document.getElementById('userAge');
    const roleInput = document.getElementById('userRole');
    const interestInput = document.getElementById('userInterest');
    
    if (!nameInput || !ageInput || !interestInput) {
        showNotification('Form elements not found', 'error');
        isProcessing = false;
        return false;
    }
    
    const name = nameInput.value.trim();
    const age = parseInt(ageInput.value.trim());
    const role = roleInput ? roleInput.value.trim() : '';
    const interest = interestInput.value.trim();
    
    if (!name || !age || !interest) {
        showNotification('Please fill in all required fields.', 'error');
        isProcessing = false;
        return false;
    }
    
    if (age < 18) {
        showNotification('You must be at least 18 years old.', 'error');
        isProcessing = false;
        return false;
    }
    
    const profileData = { name, age, role, interest };
    localStorage.setItem('profileData', JSON.stringify(profileData));
    showNotification('Profile saved successfully! 🎯', 'success');
    navigateToPage(3);
    isProcessing = false;
    return false;
}

// ========== AGENT SELECTION ==========
function selectAgent(agentType) {
    document.querySelectorAll('.agent-card').forEach(card => card.classList.remove('selected'));
    const selectedCard = document.querySelector(`[data-agent="${agentType}"]`);
    if (selectedCard) selectedCard.classList.add('selected');
    
    selectedAgent = agentType;
    const continueBtn = document.getElementById('continueToChat');
    if (continueBtn) continueBtn.disabled = false;
    
    const agentNames = {
        coding: 'Pro Level Coding Expert',
        business: 'Smart Business Consultant', 
        career: 'Professional Career Coach',
        medical: 'Simple Medical Advisor',
        emotional: 'Simple Emotional Counselor',
        technical_architect: 'Technical Architect',
        general: 'NOVA Ultra Professional AI'
    };
    
    const selectedAgentNameEl = document.getElementById('selectedAgentName');
    if (selectedAgentNameEl) {
        selectedAgentNameEl.textContent = agentNames[agentType] || 'NOVA Assistant';
    }
}

function continueToChat() {
    if (!selectedAgent) return;
    
    showLoading('🤖 Initializing your AI assistant...');
    setTimeout(() => {
        hideLoading();
        navigateToPage(4);
        initializeChatInterface();
        showNotification(`${getAgentInfo(selectedAgent).name} is ready to help! 🚀`, 'success');
    }, 1500);
}

// ========== CHAT INTERFACE ==========
function initializeChatInterface() {
    if (selectedAgent) {
        const agentInfo = getAgentInfo(selectedAgent);
        const chatAgentNameEl = document.getElementById('chatAgentName');
        const chatAgentDescEl = document.getElementById('chatAgentDesc');
        if (chatAgentNameEl) chatAgentNameEl.textContent = agentInfo.name;
        if (chatAgentDescEl) chatAgentDescEl.textContent = agentInfo.description;
    }
    
    currentSession = generateSessionId();
    loadChatHistory();
    
    // Add welcome message
    const messagesContainer = document.getElementById('chatMessages');
    if (messagesContainer && !messagesContainer.querySelector('.message')) {
        const agentInfo = getAgentInfo(selectedAgent || 'general');
        addMessageToChat('bot', `Hello! I'm ${agentInfo.name}. ${agentInfo.description}. How can I assist you today?`);
    }
    voiceModeEnabled = localStorage.getItem('nova_voice_mode') === 'true';
    if (voiceModeEnabled) {
        const voiceToggleBtn = document.getElementById('voiceToggle');
        if (voiceToggleBtn) {
            voiceToggleBtn.innerHTML = '<i class="fas fa-microphone-slash"></i>';
            voiceToggleBtn.classList.add('active');
        }
    }
}
 

function getAgentInfo(agentType) {
    const agents = {
        coding: { name: 'NOVA Coding Expert', description: 'Pro Level Programming & Development Specialist' },
        business: { name: 'NOVA Business Consultant', description: 'Smart Business Strategy & Analysis Expert' },
        career: { name: 'NOVA Career Coach', description: 'Professional Career Development Expert' },
        medical: { name: 'Dr. NOVA', description: 'Simple Health & Medical Advisory' },
        emotional: { name: 'NOVA Counselor', description: 'Simple Emotional Support & Guidance' },
        technical_architect: { name: 'NOVA Architect', description: 'Technical System Design & Architecture' },
        general: { name: 'NOVA Assistant', description: 'Ultra Professional Multi-Domain AI' }
    };
    return agents[agentType] || agents.general;
}

function toggleVoiceMode() {
    voiceModeEnabled = !voiceModeEnabled;
    const voiceToggleBtn = document.getElementById('voiceToggle');
    
    if (voiceToggleBtn) {
        if (voiceModeEnabled) {
            voiceToggleBtn.innerHTML = '<i class="fas fa-microphone-slash"></i>';
            voiceToggleBtn.title = 'Voice Mode: ON (Click to turn off)';
            voiceToggleBtn.classList.add('active');
            showNotification('Voice mode enabled - NOVA will speak responses', 'success');
        } else {
            voiceToggleBtn.innerHTML = '<i class="fas fa-microphone"></i>';
            voiceToggleBtn.title = 'Voice Mode: OFF (Click to turn on)';
            voiceToggleBtn.classList.remove('active');
            showNotification('Voice mode disabled', 'info');

        }
    }
    localStorage.setItem('nova_voice_mode', voiceModeEnabled ? 'true' : 'false');
}



// ========== EVENT LISTENERS SETUP ==========
function setupEventListeners() {
    console.log('🔗 Setting up event listeners...');
    
    // Auth buttons
    const googleSignInBtn = document.getElementById('googleSignIn');
    const githubSignInBtn = document.getElementById('githubSignIn');
    const simpleLoginBtn = document.getElementById('simpleLoginBtn');
    
    if (googleSignInBtn) {
        googleSignInBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            handleSimpleLogin();
            return false;
        });
    }
    
    if (githubSignInBtn) {
        githubSignInBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            handleSimpleLogin();
            return false;
        });
    }
    
    if (simpleLoginBtn) {
        simpleLoginBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            handleSimpleLogin();
            return false;
        });
    }

    // Profile setup
    const continueBtn = document.getElementById('continueBtn');
    if (continueBtn) {
        continueBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            handleProfileSubmit(e);
            return false;
        });
    }

    // Agent Selection
    document.querySelectorAll('.agent-card').forEach(card =>
        card.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            selectAgent(card.dataset.agent);
            return false;
        })
    );
    
    const continueToChatBtn = document.getElementById('continueToChat');
    if (continueToChatBtn) {
        continueToChatBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            continueToChat();
            return false;
        });
    }

    // Voice Toggle Button
    const voiceToggleBtn = document.getElementById('voiceToggle');
    if (voiceToggleBtn) {
        voiceToggleBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            toggleVoiceMode();
            return false;
        });
    }


    // Chat Interface
    setupChatEventListeners();

    // Floating Menu
    const floatingMenuBtn = document.getElementById('floatingMenuBtn');
    if (floatingMenuBtn) {
        floatingMenuBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            toggleFloatingMenu();
            return false;
        });
    }

    // Menu Items
    const menuItems = {
    chatMenuItem: () => navigateToPage(4),
    newChatMenuItem: () => startNewChat(),   // 👈 added
    dashboardMenuItem: () => navigateToPage(5),
    settingsMenuItem: () => navigateToPage(6),
    aboutMenuItem: () => navigateToPage(7),
    githubMenuItem: () => navigateToPage(8),
    logoutMenuItem: () => handleLogout()
};
    
    Object.entries(menuItems).forEach(([id, handler]) => {
        const element = document.getElementById(id);
        if (element) {
            element.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                handler();
                return false;
            });
        }
    });

    // Back Buttons
    const backButtons = {
        dashboardBackBtn: () => navigateToPage(4),
        settingsBackBtn: () => navigateToPage(4),
        aboutBackBtn: () => navigateToPage(4),
        githubBackBtn: () => navigateToPage(4)
    };
    
    Object.entries(backButtons).forEach(([id, handler]) => {
        const element = document.getElementById(id);
        if (element) {
            element.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                handler();
                return false;
            });
        }
    });

    // GitHub Analysis
    const analyzeRepoBtn = document.getElementById('analyzeRepoBtn');
    if (analyzeRepoBtn) {
        analyzeRepoBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            analyzeGitHubRepo();
            return false;
        });
    }

    // File Upload Modal
    const fileUploadBtn = document.getElementById('fileUploadBtn');
    const fileInput = document.getElementById('fileInput');
    const closeModal = document.getElementById('closeModal');
    const filePrompt = document.getElementById('filePrompt');
    const confirmUploadBtn = document.getElementById('confirmUploadBtn');
    
    if (fileUploadBtn && fileInput) {
        fileUploadBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            fileInput.click();
            return false;
        });
    }
    
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }
    
    if (closeModal) {
        closeModal.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            closeFileModal();
            return false;
        });
    }
    
    if (filePrompt) {
        filePrompt.addEventListener('input', validateFileUpload);
    }
    
    if (confirmUploadBtn) {
        confirmUploadBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            processFileUpload();
            return false;
        });
    }

    // 🔊 Attach beep to all floating menu items
    document.querySelectorAll('.menu-item').forEach(item => {
        item.addEventListener('click', () => {
            soundManager.playBeep("click");
        });
    });
}

// Chat Input and Send Button Event Listeners - RELOAD-FREE VERSION
function setupChatEventListeners() {
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');
    const voiceBtn = document.getElementById('voiceBtn');

    if (chatInput) {
        // Enter handling fixed here
        chatInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                e.stopPropagation();
                sendMessage();
                return false;
            }
        });

        // Auto resize textarea
        chatInput.addEventListener('input', autoResize);
    }
    
    if (sendBtn) {
        sendBtn.addEventListener('click', sendMessage);
    }
    

    if (voiceBtn) {
        voiceBtn.addEventListener('click', async (e) => {
            e.preventDefault();
            if (isRecording) {
                stopVoiceRecording();
            } else {
                await startVoiceRecording();
            }
        });
    }

}
// Separate functions for chat events to prevent conflicts
function handleChatKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        e.stopPropagation();
        sendMessage();
        return false;
    }
}

function handleSendClick(e) {
    e.preventDefault();
    e.stopPropagation();
    sendMessage();
    return false;
}

function autoResize() {
    const textarea = document.getElementById('chatInput');
    if (textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
    }
}

function getLastBotMessage() {
    const messages = document.querySelectorAll('.message.bot .message-content');
    if (messages.length > 0) {
        const lastMessage = messages[messages.length - 1];
        return lastMessage.textContent.trim();
    }
    return null;
}

// ========== FILE UPLOAD FUNCTIONS ==========
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const fileNameEl = document.getElementById('fileName');
    const fileSizeEl = document.getElementById('fileSize');
    const fileModal = document.getElementById('fileModal');
    const filePromptEl = document.getElementById('filePrompt');
    const confirmUploadBtn = document.getElementById('confirmUploadBtn');
    
    if (fileNameEl) fileNameEl.textContent = file.name;
    if (fileSizeEl) fileSizeEl.textContent = formatFileSize(file.size);
    if (fileModal) fileModal.classList.add('active');
    if (filePromptEl) {
        filePromptEl.value = '';
        filePromptEl.placeholder = `What would you like me to do with ${file.name}?`;
    }
    if (confirmUploadBtn) confirmUploadBtn.disabled = true;
}

function validateFileUpload() {
    const filePromptEl = document.getElementById('filePrompt');
    const confirmUploadBtn = document.getElementById('confirmUploadBtn');
    
    if (filePromptEl && confirmUploadBtn) {
        const prompt = filePromptEl.value.trim();
        confirmUploadBtn.disabled = !prompt;
    }
}

function closeFileModal() {
    const fileModal = document.getElementById('fileModal');
    const fileInput = document.getElementById('fileInput');
    
    if (fileModal) fileModal.classList.remove('active');
    if (fileInput) fileInput.value = '';
}

// ========== UI HELPER FUNCTIONS ==========
function addMessageToChat(sender, message, metadata = {}) {
    const messagesContainer = document.getElementById('chatMessages');
    if (!messagesContainer) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    if (sender === 'bot') {
        const agentInfo = getAgentInfo(selectedAgent || 'general');
        
        let headerInfo = `🤖 ${agentInfo.name}`;
        let metadataInfo = '';
        
        if (metadata.responseTime) {
            headerInfo += ` <span class="text-xs text-gray-500">${metadata.responseTime.toFixed(2)}s</span>`;
        }
        
        if (metadata.language || metadata.emotion) {
            metadataInfo = `<div class="text-xs text-gray-500 mt-1">`;
            if (metadata.language) metadataInfo += `Language: ${metadata.language}`;
            if (metadata.emotion) metadataInfo += ` | Emotion: ${metadata.emotion}`;
            if (metadata.conversationCount) metadataInfo += ` | Count: ${metadata.conversationCount}`;
            if (metadata.fileContextUsed) metadataInfo += ` | File Context: 📎`;
            metadataInfo += `</div>`;
        }
        
        messageDiv.innerHTML = `
            <div class="font-medium text-cyan-400 mb-1 flex items-center justify-between">
                ${headerInfo}
            </div>
            <div class="message-content">${formatMessage(message)}</div>
            ${metadataInfo}
        `;
        
        addTTSButton(messageDiv, message);
        
    } else {
        messageDiv.innerHTML = `
            <div class="font-medium text-blue-400 mb-1">
                👤 You
            </div>
            <div class="message-content">${formatMessage(message)}</div>
        `;
    }
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    if (sender === 'bot' && voiceModeEnabled) {
        const cleanText = message.replace(/<[^>]*>/g, '').replace(/\*\*/g, '').trim();
        speakText(cleanText);
    }
    
    // Smooth scroll animation
    messageDiv.style.opacity = '0';
    messageDiv.style.transform = 'translateY(20px)';
    setTimeout(() => {
        messageDiv.style.transition = 'all 0.3s ease';
        messageDiv.style.opacity = '1';
        messageDiv.style.transform = 'translateY(0)';
    }, 50);
}

function addTTSButton(messageDiv, messageText) {
    const speakBtn = document.createElement('button');
    speakBtn.className = 'ml-2 px-2 py-1 text-xs bg-purple-600 hover:bg-purple-700 text-white rounded transition-colors';
    speakBtn.innerHTML = '<i class="fas fa-volume-up mr-1"></i>Speak';
    speakBtn.title = 'Click to hear this message';
    speakBtn.onclick = async (e) => {
        e.preventDefault();
        e.stopPropagation();
        const cleanText = messageText.replace(/<[^>]*>/g, '').replace(/\*\*/g, '').trim();
        await speakText(cleanText);
        return false;
    };
    
    const messageHeader = messageDiv.querySelector('.font-medium');
    if (messageHeader) {
        messageHeader.appendChild(speakBtn);
    }
}

function formatMessage(message) {
    if (!message) return '';
    
    return message
        .replace(/\*\*(.*?)\*\*/g, '<strong class="text-cyan-300">$1</strong>')
        .replace(/\*(.*?)\*/g, '<em class="text-gray-300">$1</em>')
        .replace(/`(.*?)`/g, '<code class="bg-gray-800 px-2 py-1 rounded text-cyan-400">$1</code>')
        .replace(/```([\s\S]*?)```/g, '<pre class="bg-gray-800 p-3 rounded-lg mt-2 overflow-x-auto"><code class="text-cyan-400">$1</code></pre>')
        .replace(/\n/g, '<br>');
}

function addTypingIndicator() {
    const messagesContainer = document.getElementById('chatMessages');
    if (!messagesContainer) return null;
    
    const typingDiv = document.createElement('div');
    const typingId = 'typing-' + Date.now();
    typingDiv.id = typingId;
    typingDiv.className = 'message bot';
    
    const agentInfo = getAgentInfo(selectedAgent || 'general');
    typingDiv.innerHTML = `
        <div class="font-medium text-cyan-400 mb-1">🤖 ${agentInfo.name}</div>
        <div class="flex gap-1 items-center">
            <div class="w-2 h-2 bg-cyan-400 rounded-full animate-bounce"></div>
            <div class="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style="animation-delay: 0.2s;"></div>
            <div class="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style="animation-delay: 0.4s;"></div>
            <span class="text-gray-400 ml-2">thinking...</span>
        </div>
    `;
    
    messagesContainer.appendChild(typingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    return typingId;
}

function removeTypingIndicator(typingId) {
    if (!typingId) return;
    const typingDiv = document.getElementById(typingId);
    if (typingDiv) {
        typingDiv.remove();
    }
}

function saveChatToHistory(userMessage, botResponse) {
    const chat = {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        userMessage: userMessage,
        botResponse: botResponse,
        agent: selectedAgent || 'general',
        session: currentSession
    };
    
    chatHistory.unshift(chat);
    
    // Keep only last 50 conversations
    if (chatHistory.length > 50) {
        chatHistory = chatHistory.slice(0, 50);
    }
    
    localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
}

function loadChatHistory() {
    const saved = localStorage.getItem('chatHistory');
    if (saved) {
        try {
            chatHistory = JSON.parse(saved);
            return chatHistory;
        } catch (e) {
            chatHistory = [];
        }
    }
    return [];
}

// ========== NAVIGATION AND UI FUNCTIONS ==========
function toggleFloatingMenu() {
    const floatingMenu = document.getElementById('floatingMenu');
    if (floatingMenu) {
        floatingMenu.classList.toggle('active');
    }
}

function navigateToPage(pageNumber) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    
    // Show target page
    const targetPage = document.getElementById(`page${pageNumber}`);
    if (targetPage) {
        targetPage.classList.add('active');
        currentPage = pageNumber;
    }
    
    // Close floating menu
    const floatingMenu = document.getElementById('floatingMenu');
    if (floatingMenu) {
        floatingMenu.classList.remove('active');
    }
    
    // Initialize chat interface if navigating to chat
    if (pageNumber === 4) {
        setTimeout(() => initializeChatInterface(), 100);
    }
    
    console.log(`📍 Navigated to page ${pageNumber}`);
}

function showLoading(message = 'Loading...') {
    const overlay = document.getElementById('loadingOverlay');
    const messageEl = document.getElementById('loadingMessage');
    
    if (overlay) {
        overlay.classList.remove('hidden');
    }
    
    if (messageEl) {
        messageEl.textContent = message;
    }
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.classList.add('hidden');
    }
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `fixed top-4 right-4 z-50 p-4 rounded-lg shadow-lg max-w-sm transition-all duration-300 transform translate-x-full`;
    
    const colors = {
        info: 'bg-blue-600 border-blue-500',
        success: 'bg-green-600 border-green-500',
        error: 'bg-red-600 border-red-500',
        warning: 'bg-yellow-600 border-yellow-500'
    };
    
    const icons = {
        info: 'info-circle',
        success: 'check-circle', 
        error: 'exclamation-circle',
        warning: 'exclamation-triangle'
    };
    
    notification.className += ` ${colors[type] || colors.info} border-l-4 text-white`;
    notification.innerHTML = `
        <div class="flex items-center">
            <i class="fas fa-${icons[type] || icons.info} mr-2"></i>
            <span>${message}</span>
            <button class="ml-3 text-white hover:text-gray-200" onclick="this.parentElement.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.classList.remove('translate-x-full');
    }, 100);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (document.body.contains(notification)) {
            notification.classList.add('translate-x-full');
            setTimeout(() => {
                if (document.body.contains(notification)) {
                    document.body.removeChild(notification);
                }
            }, 300);
        }
    }, 5000);
}

// ========== UTILITY FUNCTIONS ==========
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function isValidGitHubUrl(url) {
    return /^https:\/\/github\.com\/[a-zA-Z0-9_.-]+\/[a-zA-Z0-9_.-]+(\/)?(\?.*)?$/.test(url);
}

function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

function createStars() {
    const starsContainer = document.getElementById('stars');
    if (starsContainer && starsContainer.children.length === 0) {
        for (let i = 0; i < 100; i++) {
            const star = document.createElement('div');
            star.className = 'star';
            star.style.left = Math.random() * 100 + '%';
            star.style.top = Math.random() * 100 + '%';
            star.style.animationDelay = Math.random() * 3 + 's';
            starsContainer.appendChild(star);
        }
    }
}

// ========== SYSTEM INITIALIZATION ==========
async function initializeSystemCheck() {
    console.log('🔍 Checking backend connection...');
    
    try {
        // Check all backend endpoints
        const [systemInfo, health, status] = await Promise.all([
            getSystemInfo(),
            healthCheck(),
            getSystemStatus()
        ]);
        
        if (health && systemInfo) {
            console.log('✅ NOVA Backend Connected Successfully');
            console.log(`🤖 Backend Version: ${systemInfo.version}`);
            console.log(`📊 Status: ${systemInfo.status}`);
            console.log(`🎯 Features: ${systemInfo.features.length} available`);
            
            showNotification('NOVA Backend Connected! 🚀', 'success');
            
            if (status) {
                console.log('📈 System Status Loaded');
                console.log(`💥 Active Sessions: ${status.session_info?.total_sessions || 0}`);
                console.log(`🔗 API Providers: ${status.session_info?.available_providers || 0}`);
            }
            
        } else {
            throw new Error('Backend connection failed');
        }
        
    } catch (error) {
        console.warn('⚠️ Backend connection failed:', error);
        showNotification(`Backend may not be running. Please start the server at ${API_BASE}`, 'warning');
    }
}

// ========== ENHANCED ERROR HANDLING ==========
window.addEventListener('error', function(event) {
    console.error('🚨 Global Error:', event.error);
    if (event.error && event.error.message.includes('fetch')) {
        showNotification('Connection error. Please check if the backend server is running.', 'error');
    }
});

window.addEventListener('unhandledrejection', function(event) {
    console.error('🚨 Unhandled Promise Rejection:', event.reason);
    if (event.reason && event.reason.message && event.reason.message.includes('fetch')) {
        showNotification('API request failed. Please check backend connection.', 'error');
    }
});

// ========== FINAL INITIALIZATION ==========
window.addEventListener('load', function() {
    console.log('🚀 NOVA Ultra Professional Frontend Initialized');
    console.log('🔐 Authentication: Demo Mode (user@gmail.com / password123)');
    console.log('🎯 Backend API:', API_BASE);
    console.log('💡 All backend endpoints integrated with exact FastAPI logic matching');
    console.log('👤 User ID:', userId);
    
    // Global form submission prevention (enhanced)
    document.addEventListener('submit', function(e) {
        console.log('🛑 Form submission prevented globally');
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        return false;
    }, true);
    
    // Enhanced Enter key handling
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && e.target.tagName === 'INPUT' && e.target.id !== 'chatInput') {
            // Allow Enter in regular inputs but prevent form submission
            const form = e.target.closest('form');
            if (form) {
                e.preventDefault();
                e.stopPropagation();
                return false;
            }
        }
    });
    
    // Initialize system check
    setTimeout(() => {
        initializeSystemCheck();
    }, 1000);
    
    // Add development helper
    window.NOVA = {
        // Core API functions
        sendMessage,
        clearContext,
        getSystemStatus,
        healthCheck,
        getSystemInfo,
        performWebSearch,
        processFileUpload,
        analyzeGitHubRepo,
        askGitHubQuestion,
        speakText,
        
        // Navigation and UI
        navigateToPage,
        showNotification,
        addMessageToChat,
        
        // State and config
        API_BASE,
        userId,
        currentSession,
        selectedAgent,
        chatHistory,
        
        // Utility functions
        formatFileSize,
        isValidGitHubUrl,
        generateSessionId
    };
    
    console.log('🎉 NOVA Frontend Ready! Access helper functions via window.NOVA');
    console.log('📚 Available NOVA functions:', Object.keys(window.NOVA));
    
});