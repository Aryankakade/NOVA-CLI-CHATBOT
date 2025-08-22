// ========== NOVA ULTRA PROFESSIONAL - COMPLETE REPLIT_BACKEND.PY INTEGRATION ==========
// EXACT BACKEND CONFIGURATION - MATCHES replit_backend.py
const API_BASE = "http://127.0.0.1:8000"; // Enhanced Backend API
console.log('🔗 NOVA Frontend connecting to Enhanced Backend:', API_BASE);

// ========== GLOBAL VARIABLES ==========
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
let systemInfo = null;
let mlSystemStatus = null;
const VOICE_PROCESS_ENDPOINT = `${API_BASE}/voice/process`;

// ========== COMPLETE FORM SUBMISSION PREVENTION ==========
document.addEventListener('DOMContentLoaded', function () {
    console.log('🚀 NOVA Ultra Professional Frontend with Complete Enhanced Integration Starting...');
    
    // Prevent ALL form submissions globally
    document.addEventListener('submit', function (e) {
        e.preventDefault();
    });

    // Handle Enter key inside chatInput
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.addEventListener('keydown', function (e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault(); // stop page reload
                e.stopPropagation();
                sendMessage();
                return false;
            }
        });
    }

    // Initialize app
    initializeApp();
});

function initializeApp() {
    console.log('🔧 Initializing Complete Enhanced NOVA System...');
    console.log('🔗 Backend API:', API_BASE);
    
    // Initialize user ID
    userId = localStorage.getItem('nova_user_id') || `web-user-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    localStorage.setItem('nova_user_id', userId);
    
    createStars();
    setupEventListeners();
    checkLoginState();
    initializeSystemCheck();
}

// ========== ENHANCED BACKEND SYSTEM CHECK ==========
async function initializeSystemCheck() {
    console.log('🔍 Checking Enhanced Backend System...');
    
    try {
        showSystemStatus('Connecting to Enhanced Backend...', 'checking');
        
        // Check system info from replit_backend.py
        const systemData = await getSystemInfo();
        
        if (systemData) {
            console.log('✅ Enhanced Backend System Connected:', systemData);
            systemInfo = systemData;
            
            // Check ML system status
            await checkMLSystemStatus();
            
            showSystemStatus('Enhanced Backend Connected ✅', 'connected');
            updateConnectionIndicator(true);
            
        } else {
            throw new Error('Failed to connect to enhanced backend');
        }
        
    } catch (error) {
        console.error('❌ Enhanced Backend Connection Failed:', error);
        showSystemStatus('Backend Connection Failed ❌', 'error');
        updateConnectionIndicator(false);
        showNotification('Failed to connect to enhanced backend. Please ensure replit_backend.py is running.', 'error');
    }
}

async function checkMLSystemStatus() {
    try {
        // Check if ML system is available from backend
        const mlStatus = await fetch(`${API_BASE}/system/ml-status`).catch(() => null);
        
        if (mlStatus && mlStatus.ok) {
            const mlData = await mlStatus.json();
            mlSystemStatus = mlData;
            showMLStatus('Enhanced ML Pipeline Active ✅', 'active');
            console.log('🧠 ML System Status:', mlData);
        } else {
            showMLStatus('ML Pipeline: Smart Mode', 'basic');
            console.log('ℹ️ ML System: Running in smart mode');
        }
        
    } catch (error) {
        console.log('ℹ️ ML System check failed, using smart mode');
        showMLStatus('ML Pipeline: Smart Mode', 'basic');
    }
}

function showSystemStatus(message, status) {
    const statusElement = document.getElementById('statusText');
    const statusDot = document.querySelector('.status-dot');
    
    if (statusElement) statusElement.textContent = message;
    
    if (statusDot) {
        statusDot.className = `fas fa-circle status-dot status-${status}`;
    }
}

function showMLStatus(message, status) {
    const mlStatusElement = document.getElementById('mlStatusText');
    const mlIndicator = document.getElementById('mlIndicator');
    
    if (mlStatusElement) mlStatusElement.textContent = message;
    
    if (mlIndicator) {
        mlIndicator.className = `indicator ml-${status}`;
    }
}

function updateConnectionIndicator(connected) {
    const indicator = document.getElementById('connectionIndicator');
    if (indicator) {
        indicator.className = `indicator ${connected ? 'connected' : 'disconnected'}`;
        const span = indicator.querySelector('span');
        if (span) span.textContent = connected ? 'Connected' : 'Disconnected';
    }
}

// ========== COMPLETE BACKEND API INTEGRATION - ALL ENDPOINTS ==========

// EXACT BACKEND INTEGRATION: GET / endpoint
async function getSystemInfo() {
    try {
        console.log('📡 Fetching system info from enhanced backend...');
        
        const response = await fetch(`${API_BASE}/`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json'
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            console.log('✅ Enhanced System Info:', data);
            return data;
        } else {
            console.error('❌ System info request failed:', response.status, response.statusText);
            return null;
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
        console.log(`🤖 Sending message to enhanced backend: "${message}"`);
        console.log(`🎯 Using agent: ${selectedAgent}`);
        
        // EXACT backend POST /chat endpoint with ChatRequest model
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                user_id: userId,
                agent_type: selectedAgent
            })
        });

        if (!response.ok) {
            throw new Error(`Enhanced Backend Error: ${response.status} - ${response.statusText}`);
        }

        const data = await response.json();
        console.log('✅ Enhanced Backend Response:', data);
        
        removeTypingIndicator(typingId);
        
        // Process ChatResponse model response - Complete integration
        const botResponse = data.response || 'No response received from enhanced system.';
        const agentUsed = data.agent_used || selectedAgent || 'general';
        const responseTime = data.response_time || 0;
        const language = data.language || 'english';
        const emotion = data.emotion || 'neutral';
        const emotionConfidence = data.emotion_confidence || 0.0;
        const agentConfidence = data.agent_confidence || 0.0;
        const conversationCount = data.conversation_count || 0;
        const fileContextUsed = data.file_context_used || false;
        const sessionId = data.session_id || 'unknown';
        
        // Enhanced ML information from replit_backend.py
        const mlEnhanced = data.ml_enhanced || false;
        const enhancementReason = data.enhancement_reason || '';
        const queryComplexity = data.query_complexity || 'simple';
        
        // Show enhancement indicator
        updateEnhancementStatus(mlEnhanced, enhancementReason, queryComplexity);
        
        addMessageToChat('bot', botResponse, { 
            agent: agentUsed, 
            responseTime: responseTime,
            language: language,
            emotion: emotion,
            emotionConfidence: emotionConfidence,
            agentConfidence: agentConfidence,
            conversationCount: conversationCount,
            fileContextUsed: fileContextUsed,
            sessionId: sessionId,
            mlEnhanced: mlEnhanced,
            enhancementReason: enhancementReason,
            queryComplexity: queryComplexity
        });

        // 🔊 AI responded → success beep
        soundManager.playBeep("success");
        
        saveChatToHistory(message, botResponse);
        
        // Update selected agent if backend switched agents
        if (agentUsed !== selectedAgent) {
            selectedAgent = agentUsed;
            updateAgentDisplay(agentUsed);
            showNotification(`Switched to ${getAgentInfo(agentUsed).name} 🔄`, 'info');
        }

        console.log('📊 Enhanced Response Stats:', {
            agent: agentUsed,
            language: language,
            emotion: emotion,
            responseTime: responseTime,
            conversationCount: conversationCount,
            fileContextUsed: fileContextUsed,
            mlEnhanced: mlEnhanced,
            queryComplexity: queryComplexity
        });

    } catch (err) {
        console.error('❌ Enhanced Chat Error:', err);
        removeTypingIndicator(typingId);
        addMessageToChat('bot', `❌ Enhanced Backend Error: ${err.message}. Please ensure replit_backend.py is running on ${API_BASE}`);

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
        console.log(`📎 Uploading file to enhanced backend: ${file.name}`);
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('user_id', userId);
        if (prompt) formData.append('prompt', prompt);

        const response = await fetch(`${API_BASE}/file/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Enhanced file upload error: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('✅ Enhanced File Processing Response:', data);
        
        removeTypingIndicator(typingId);

        if (data.success && data.metadata?.file_analysis) {
            const analysis = data.metadata.file_analysis;

            let responseText = `📄 **File Analysis Complete!**\n\n`;
            responseText += `**File:** ${analysis.file_name}\n`;
            responseText += `**Type:** ${analysis.file_type}\n`;
            responseText += `**Size:** ${formatFileSize(analysis.file_size)}\n`;
            if (analysis.lines) responseText += `**Lines:** ${analysis.lines}\n`;
            if (analysis.words) responseText += `**Words:** ${analysis.words}\n`;
            if (analysis.chars) responseText += `**Characters:** ${analysis.chars}\n`;

            // ✅ Show AI's answer (from standardized field)
            responseText += `\n🤖 **Enhanced AI Analysis:**\n${data.response || 'No AI response generated.'}`;
            
            // ML enhancement info
            if (data.ml_enhanced) {
                responseText += `\n\n🧠 **ML Enhancement Applied:** ${data.enhancement_reason}`;
            }
            
            addMessageToChat('bot', responseText);
            soundManager.playBeep("success");

        } else {
            throw new Error(data.error || data.message || 'Enhanced file processing failed');
        }
        
        saveChatToHistory(
            `[File: ${file.name}] ${displayMessage}`,
            data.response || 'File processed successfully with enhanced analysis'
        );
        showNotification('File processed with enhanced analysis! 📄', 'success');
        
    } catch (error) {
        console.error('❌ Enhanced file processing error:', error);
        removeTypingIndicator(typingId);
        addMessageToChat('bot', `❌ Enhanced file processing failed: ${error.message}`);
        soundManager.playBeep("error");
        showNotification('Enhanced file processing failed: ' + error.message, 'error');
    }
}

// EXACT BACKEND INTEGRATION: POST /voice/process endpoint
async function speakText(text) {
    if (!text) return false;
    
    try {
        showNotification('🔊 Generating enhanced speech...', 'info');
        
        const formData = new FormData();
        formData.append('text', text);
        formData.append('user_id', userId);

        const response = await fetch(VOICE_PROCESS_ENDPOINT, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Enhanced TTS error: ${response.status}`);
        }
        
        // Get audio blob and play it
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        
        audio.onloadeddata = () => {
            showNotification('🔊 Playing enhanced audio...', 'success');
            soundManager.playBeep("success");   // ✅ double beep when audio is ready
        };
        
        audio.onended = () => {
            URL.revokeObjectURL(audioUrl);
            showNotification('🔊 Enhanced audio complete', 'success');
            soundManager.playBeep("success");   // ✅ beep after playback ends
        };
        
        audio.onerror = () => {
            showNotification('❌ Enhanced audio playback failed', 'error');
            soundManager.playBeep("error");     // ❌ error beep
        };
        
        await audio.play();
        return true;
        
    } catch (error) {
        console.error('Enhanced TTS error:', error);
        showNotification('Enhanced TTS error: ' + error.message, 'error');
        soundManager.playBeep("error");         // ❌ error beep
        return false;
    }
}

// ========== VOICE PROCESSING ==========
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
    showLoading('Processing voice with enhanced ML...');
    
    try {
        // Combine audio chunks into a single blob
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm;codecs=opus' });
        
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.webm');
        formData.append('user_id', userId);

        const response = await fetch(VOICE_PROCESS_ENDPOINT, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Enhanced voice processing error: ${response.status}`);
        }
        
        const data = await response.json();
        hideLoading();
        
        if (data.success && data.transcribed_text) {
            const transcribedText = data.transcribed_text;
            console.log('🎤 Enhanced voice transcription:', transcribedText);
            
            // Put transcribed text in input and send
            const chatInput = document.getElementById('chatInput');
            if (chatInput) {
                chatInput.value = transcribedText;
                autoResize();
                sendMessage();
            }
            
            soundManager.playBeep("success");
            showNotification('Voice processed with enhanced ML! 🎤', 'success');
            
        } else {
            throw new Error(data.error || 'Enhanced voice processing failed');
        }
        
    } catch (error) {
        console.error('❌ Enhanced voice processing error:', error);
        hideLoading();
        showNotification('Enhanced voice processing failed: ' + error.message, 'error');
        soundManager.playBeep("error");
    }
}

function updateVoiceButtonState() {
    const voiceBtn = document.getElementById('voiceRecordBtn');
    if (!voiceBtn) return;
    
    const icon = voiceBtn.querySelector('i');
    
    if (isRecording) {
        voiceBtn.classList.add('recording');
        if (icon) icon.className = 'fas fa-stop';
        voiceBtn.title = 'Stop Recording';
    } else {
        voiceBtn.classList.remove('recording');
        if (icon) icon.className = 'fas fa-microphone';
        voiceBtn.title = 'Voice Input';
    }
}

// ========== ENHANCEMENT STATUS DISPLAY ==========
function updateEnhancementStatus(mlEnhanced, reason, complexity) {
    const enhancementStatus = document.getElementById('enhancementStatus');
    if (!enhancementStatus) return;
    
    const icon = enhancementStatus.querySelector('i');
    const text = enhancementStatus.querySelector('span');
    
    if (mlEnhanced) {
        enhancementStatus.className = 'enhancement-status enhanced';
        if (icon) icon.className = 'fas fa-brain';
        if (text) text.textContent = `ML Enhanced: ${reason}`;
        
        // Auto-reset after 5 seconds
        setTimeout(() => {
            enhancementStatus.className = 'enhancement-status';
            if (text) text.textContent = 'Smart Enhancement Ready';
        }, 5000);
    } else {
        enhancementStatus.className = 'enhancement-status simple';
        if (icon) icon.className = 'fas fa-flash';
        if (text) text.textContent = `Quick Response: ${complexity}`;
        
        // Auto-reset after 3 seconds
        setTimeout(() => {
            enhancementStatus.className = 'enhancement-status';
            if (text) text.textContent = 'Smart Enhancement Ready';
        }, 3000);
    }
}

// ========== COMPLETE SPECIAL COMMANDS SYSTEM ==========
async function handleSpecialCommands(message) {
    const command = message.toLowerCase().trim();
    
    // System commands
    if (command === '/system' || command === '/status') {
        const systemData = await getSystemInfo();
        if (systemData) {
            let statusMessage = `🔧 **Enhanced System Status**\n\n`;
            statusMessage += `**Status:** ${systemData.status}\n`;
            statusMessage += `**Version:** ${systemData.version}\n`;
            statusMessage += `**Enhanced Features:** ${systemData.features ? systemData.features.join(', ') : 'Smart Detection'}\n`;
            statusMessage += `**ML Pipeline:** ${mlSystemStatus ? 'Enhanced Active' : 'Smart Mode'}\n`;
            statusMessage += `**Backend:** replit_backend.py\n`;
            statusMessage += `**User ID:** ${userId}\n`;
            
            addMessageToChat('bot', statusMessage);
        } else {
            addMessageToChat('bot', '❌ Could not retrieve enhanced system status');
        }
        return true;
    }
    
    // Clear chat
    if (command === '/clear') {
        clearChat();
        return true;
    }
    
    // Help command
    if (command === '/help') {
        const helpMessage = `🆘 **Enhanced NOVA Commands**\n\n` +
            `**/system** - Enhanced system status\n` +
            `**/clear** - Clear chat history\n` +
            `**/help** - Show this help\n` +
            `**/agents** - List available agents\n` +
            `**/analyze <repo_url>** - Analyze GitHub repository\n` +
            `**/ask <question> github** - Ask GitHub repository question\n\n` +
            `**Enhanced Features:**\n` +
            `• Smart ML Enhancement Detection\n` +
            `• Advanced Query Processing\n` +
            `• Professional Agent Routing\n` +
            `• File Analysis & Processing\n` +
            `• Voice Input/Output\n` +
            `• GitHub Integration\n` +
            `• Contextual Memory System`;
        
        addMessageToChat('bot', helpMessage);
        return true;
    }
    
    // Agents command
    if (command === '/agents') {
        const agentsMessage = `👥 **Available Professional Agents**\n\n` +
            `🧠 **General AI** - Smart assistance with ML enhancement\n` +
            `💻 **Coding Expert** - Advanced programming assistance\n` +
            `💼 **Career Coach** - Professional development guidance\n` +
            `📊 **Business Consultant** - Strategic business insights\n` +
            `🏥 **Health Advisor** - Medical information and wellness\n` +
            `💙 **Emotional Support** - Mental health and counseling (Always AI responses)\n` +
            `🏗️ **Technical Architect** - System design and architecture\n\n` +
            `**Note:** Emotional and Health agents provide AI responses even for casual conversations!`;
        
        addMessageToChat('bot', agentsMessage);
        return true;
    }
    
    // GitHub analysis
    if (command.startsWith('/analyze ') && command.includes('github')) {
        const repoUrl = message.substring(9).trim();
        const result = await analyzeGitHubRepo(repoUrl);
        addMessageToChat('bot', `**GitHub Repository Analysis:**\n\n${result}`);
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

// ========== GITHUB INTEGRATION SYSTEM ==========
async function analyzeGitHubRepo(repoUrl) {
    try {
        showLoading('Analyzing GitHub repository...');
        
        const response = await fetch(`${API_BASE}/github/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                repo_url: repoUrl,
                user_id: userId
            })
        });
        
        hideLoading();
        
        if (response.ok) {
            const data = await response.json();
            return data.analysis || 'Repository analysis completed successfully.';
        } else {
            throw new Error(`GitHub analysis failed: ${response.status}`);
        }
        
    } catch (error) {
        hideLoading();
        return `GitHub analysis error: ${error.message}`;
    }
}

async function askGitHubQuestion(question) {
    try {
        showLoading('Searching GitHub repository...');
        
        const response = await fetch(`${API_BASE}/github/qa`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: question,
                user_id: userId
            })
        });
        
        hideLoading();
        
        if (response.ok) {
            const data = await response.json();
            return data.answer || 'No answer found in the repository.';
        } else {
            throw new Error(`GitHub search failed: ${response.status}`);
        }
        
    } catch (error) {
        hideLoading();
        return `GitHub search error: ${error.message}`;
    }
}

// ========== AUTHENTICATION SYSTEM ==========
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

// ========== SOUND MANAGER ==========
class SoundManager {
    constructor() {
        this.enabled = true;
    }

    playBeep(type) {
        if (!this.enabled) return;

        try {
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
        } catch (e) {
            console.log('Audio context failed:', e);
        }
    }
}

// Global instance
const soundManager = new SoundManager();

// ========== AGENT MANAGEMENT ==========
function getAgentInfo(agentType) {
    const agents = {
        general: { name: 'General AI Assistant', description: 'Smart assistance with ML enhancement when needed' },
        coding: { name: 'Pro Level Coding Expert', description: 'Advanced programming and development assistance' },
        career: { name: 'Professional Career Coach', description: 'Professional development and career guidance' },
        business: { name: 'Smart Business Consultant', description: 'Strategic business insights and analysis' },
        medical: { name: 'Simple Medical Advisor', description: 'Medical information and wellness guidance (Always AI responses)' },
        emotional: { name: 'Simple Emotional Counselor', description: 'Mental health and emotional counseling (Always AI responses)' },
        technical_architect: { name: 'Technical Architect', description: 'System design and technical architecture' }
    };
    
    return agents[agentType] || agents.general;
}

function updateAgentDisplay(agentType) {
    const agentInfo = getAgentInfo(agentType);
    const chatAgentNameEl = document.getElementById('chatAgentName');
    const chatAgentDescEl = document.getElementById('chatAgentDesc');
    
    if (chatAgentNameEl) chatAgentNameEl.textContent = agentInfo.name;
    if (chatAgentDescEl) chatAgentDescEl.textContent = agentInfo.description;
    
    // Update agent cards selection
    document.querySelectorAll('.agent-card').forEach(card => {
        card.classList.remove('selected');
    });
    
    const selectedCard = document.querySelector(`[data-agent="${agentType}"]`);
    if (selectedCard) {
        selectedCard.classList.add('selected');
    }
}

function selectAgent(agentType) {
    document.querySelectorAll('.agent-card').forEach(card => card.classList.remove('selected'));
    const selectedCard = document.querySelector(`[data-agent="${agentType}"]`);
    if (selectedCard) selectedCard.classList.add('selected');
    
    selectedAgent = agentType;
    const continueBtn = document.getElementById('continueToChat');
    if (continueBtn) continueBtn.disabled = false;
    
    const agentInfo = getAgentInfo(agentType);
    const selectedAgentNameEl = document.getElementById('selectedAgentName');
    if (selectedAgentNameEl) {
        selectedAgentNameEl.textContent = agentInfo.name;
    }
    
    updateAgentDisplay(agentType);
    showNotification(`Selected ${agentInfo.name} 🔄`, 'info');
    soundManager.playBeep("click");
}

function continueToChat() {
    if (!selectedAgent) {
        showNotification('Please select an agent first', 'error');
        return;
    }
    
    navigateToPage(4);
    initializeChatInterface();
    
    const agentInfo = getAgentInfo(selectedAgent);
    showNotification(`Ready to chat with ${agentInfo.name}! 🎯`, 'success');
}

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

// ========== NAVIGATION ==========
function navigateToPage(pageNumber) {
    currentPage = pageNumber;
    
    // Hide all pages
    for (let i = 1; i <= 4; i++) {
        const page = document.getElementById(`page${i}`);
        if (page) page.style.display = 'none';
    }
    
    // Show current page
    const currentPageEl = document.getElementById(`page${pageNumber}`);
    if (currentPageEl) currentPageEl.style.display = 'flex';
    
    // Update page-specific content
    if (pageNumber === 4 && selectedAgent) {
        initializeChatInterface();
    }
}

function initializeChatInterface() {
    // Initialize chat with selected agent
    if (selectedAgent) {
        updateAgentDisplay(selectedAgent);
        
        const agentInfo = getAgentInfo(selectedAgent);
        const welcomeMessage = `Hello! I'm ${agentInfo.name}. ${agentInfo.description} How can I assist you today?`;
        
        // Clear and initialize chat
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages) {
            chatMessages.innerHTML = `
                <div class="welcome-message">
                    <div class="message-content">
                        <h3>Welcome to ${agentInfo.name}!</h3>
                        <p>${agentInfo.description}</p>
                        <p><strong>Enhanced Features:</strong> Smart ML detection ensures you get the right level of AI assistance for your queries.</p>
                    </div>
                </div>
            `;
            
            addMessageToChat('bot', welcomeMessage);
        }
    }
}

// ========== CHAT MANAGEMENT ==========
function clearChat() {
    const chatMessages = document.getElementById('chatMessages');
    if (chatMessages) {
        if (selectedAgent) {
            const agentInfo = getAgentInfo(selectedAgent);
            chatMessages.innerHTML = `
                <div class="welcome-message">
                    <div class="message-content">
                        <h3>Chat Cleared - ${agentInfo.name}</h3>
                        <p>${agentInfo.description}</p>
                        <p><strong>Enhanced Features:</strong> Smart ML detection ensures optimal responses for your queries.</p>
                    </div>
                </div>
            `;
        } else {
            chatMessages.innerHTML = `
                <div class="welcome-message">
                    <div class="message-content">
                        <h3>Welcome to NOVA Ultra Professional!</h3>
                        <p>Your advanced AI assistant with enhanced ML capabilities.</p>
                    </div>
                </div>
            `;
        }
    }
    
    chatHistory = [];
    showNotification('Chat cleared', 'info');
    soundManager.playBeep("click");
}

function startNewChat() {
    clearChat();
    const floatingMenu = document.getElementById('floatingMenu');
    if (floatingMenu) {
        floatingMenu.classList.remove('active');
    }
}

function addMessageToChat(sender, message, metadata = {}) {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    const messageText = document.createElement('div');
    messageText.className = 'message-text';
    messageText.innerHTML = formatMessage(message);
    
    messageContent.appendChild(messageText);
    
    // Add metadata for bot messages
    if (sender === 'bot' && Object.keys(metadata).length > 0) {
        const metadataDiv = document.createElement('div');
        metadataDiv.className = 'message-metadata';
        
        if (metadata.agent) {
            const agentTag = document.createElement('span');
            agentTag.className = 'metadata-item';
            agentTag.innerHTML = `<i class="fas fa-robot"></i> ${metadata.agent}`;
            metadataDiv.appendChild(agentTag);
        }
        
        if (metadata.mlEnhanced !== undefined) {
            const enhancementTag = document.createElement('span');
            enhancementTag.className = `metadata-item ${metadata.mlEnhanced ? 'enhanced' : 'simple'}`;
            enhancementTag.innerHTML = metadata.mlEnhanced 
                ? `<i class="fas fa-brain"></i> ML Enhanced`
                : `<i class="fas fa-flash"></i> Quick Response`;
            metadataDiv.appendChild(enhancementTag);
        }
        
        if (metadata.responseTime) {
            const timeTag = document.createElement('span');
            timeTag.className = 'metadata-item';
            timeTag.innerHTML = `<i class="fas fa-clock"></i> ${metadata.responseTime.toFixed(2)}s`;
            metadataDiv.appendChild(timeTag);
        }
        
        if (metadata.emotion && metadata.emotion !== 'neutral') {
            const emotionTag = document.createElement('span');
            emotionTag.className = 'metadata-item';
            emotionTag.innerHTML = `<i class="fas fa-heart"></i> ${metadata.emotion}`;
            metadataDiv.appendChild(emotionTag);
        }
        
        messageContent.appendChild(metadataDiv);
        
        // Add action buttons
        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'message-actions';
        
        const copyBtn = document.createElement('button');
        copyBtn.className = 'action-btn';
        copyBtn.innerHTML = '<i class="fas fa-copy"></i>';
        copyBtn.title = 'Copy message';
        copyBtn.onclick = () => copyToClipboard(message);
        
        const speakBtn = document.createElement('button');
        speakBtn.className = 'action-btn';
        speakBtn.innerHTML = '<i class="fas fa-volume-up"></i>';
        speakBtn.title = 'Speak message';
        speakBtn.onclick = () => speakText(message);
        
        actionsDiv.appendChild(copyBtn);
        actionsDiv.appendChild(speakBtn);
        messageContent.appendChild(actionsDiv);
    }
    
    messageDiv.appendChild(messageContent);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addTypingIndicator() {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return null;
    
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message typing-indicator';
    typingDiv.id = `typing-${Date.now()}`;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    const typingAnimation = document.createElement('div');
    typingAnimation.className = 'typing-animation';
    typingAnimation.innerHTML = '<span></span><span></span><span></span>';
    
    messageContent.appendChild(typingAnimation);
    typingDiv.appendChild(messageContent);
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return typingDiv.id;
}

function removeTypingIndicator(typingId) {
    if (typingId) {
        const typingElement = document.getElementById(typingId);
        if (typingElement) {
            typingElement.remove();
        }
    }
}

function formatMessage(message) {
    // Convert markdown-like formatting
    message = message.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    message = message.replace(/\*(.*?)\*/g, '<em>$1</em>');
    message = message.replace(/`(.*?)`/g, '<code>$1</code>');
    
    // Convert line breaks
    message = message.replace(/\n/g, '<br>');
    
    // Convert bullet points
    message = message.replace(/^• /gm, '• ');
    
    return message;
}

function saveChatToHistory(userMessage, botResponse) {
    chatHistory.push({
        user: userMessage,
        bot: botResponse,
        timestamp: new Date().toISOString(),
        agent: selectedAgent
    });
    
    // Keep only last 50 messages
    if (chatHistory.length > 50) {
        chatHistory = chatHistory.slice(-50);
    }
    
    localStorage.setItem('nova_chat_history', JSON.stringify(chatHistory));
}

// ========== EVENT LISTENERS ==========
function setupEventListeners() {
    // Login functionality
    const simpleLoginBtn = document.getElementById('simpleLoginBtn');
    if (simpleLoginBtn) {
        simpleLoginBtn.addEventListener('click', handleSimpleLogin);
    }
    
    // Profile functionality
    const profileForm = document.getElementById('profileForm');
    if (profileForm) {
        profileForm.addEventListener('submit', handleProfileSubmit);
    }
    
    const profileSubmitBtn = document.getElementById('profileSubmit');
    if (profileSubmitBtn) {
        profileSubmitBtn.addEventListener('click', handleProfileSubmit);
    }
    
    // Chat functionality
    const sendBtn = document.getElementById('sendBtn');
    if (sendBtn) {
        sendBtn.addEventListener('click', sendMessage);
    }
    
    // Voice functionality
    const voiceRecordBtn = document.getElementById('voiceRecordBtn');
    if (voiceRecordBtn) {
        voiceRecordBtn.addEventListener('click', toggleVoiceRecording);
    }
    
    // File upload functionality
    const fileUploadBtn = document.getElementById('fileUploadBtn');
    if (fileUploadBtn) {
        fileUploadBtn.addEventListener('click', openFileModal);
    }
    
    const processFileBtn = document.getElementById('processFile');
    if (processFileBtn) {
        processFileBtn.addEventListener('click', processFileUpload);
    }
    
    // Modal functionality
    const closeFileModalBtn = document.getElementById('closeFileModal');
    if (closeFileModalBtn) {
        closeFileModalBtn.addEventListener('click', closeFileModal);
    }
    
    const cancelFileUploadBtn = document.getElementById('cancelFileUpload');
    if (cancelFileUploadBtn) {
        cancelFileUploadBtn.addEventListener('click', closeFileModal);
    }
    
    // Chat controls
    const clearChatBtn = document.getElementById('clearChat');
    if (clearChatBtn) {
        clearChatBtn.addEventListener('click', clearChat);
    }
    
    const newChatBtn = document.getElementById('newChat');
    if (newChatBtn) {
        newChatBtn.addEventListener('click', startNewChat);
    }
    
    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', handleLogout);
    }
    
    // Navigation
    const continueToChat = document.getElementById('continueToChat');
    if (continueToChat) {
        continueToChat.addEventListener('click', continueToChat);
    }
    
    // Agent selection
    document.querySelectorAll('.agent-card').forEach(card => {
        card.addEventListener('click', function() {
            const agentType = this.getAttribute('data-agent');
            selectAgent(agentType);
        });
    });
    
    // Auto-resize chat input
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.addEventListener('input', autoResize);
        chatInput.addEventListener('paste', () => setTimeout(autoResize, 0));
    }
    
    // File drag and drop
    const fileUploadArea = document.getElementById('fileUploadArea');
    const fileInput = document.getElementById('fileInput');
    
    if (fileUploadArea && fileInput) {
        fileUploadArea.addEventListener('click', () => fileInput.click());
        
        fileUploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.classList.add('dragover');
        });
        
        fileUploadArea.addEventListener('dragleave', function(e) {
            e.preventDefault();
            this.classList.remove('dragover');
        });
        
        fileUploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            this.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                showFilePreview(files[0]);
            }
        });
        
        fileInput.addEventListener('change', function() {
            if (this.files.length > 0) {
                showFilePreview(this.files[0]);
            }
        });
    }
    
    // Floating menu
    const floatingMenuBtn = document.getElementById('floatingMenuBtn');
    const floatingMenu = document.getElementById('floatingMenu');
    
    if (floatingMenuBtn && floatingMenu) {
        floatingMenuBtn.addEventListener('click', function() {
            floatingMenu.classList.toggle('active');
        });
        
        // Close menu when clicking outside
        document.addEventListener('click', function(e) {
            if (!floatingMenu.contains(e.target) && !floatingMenuBtn.contains(e.target)) {
                floatingMenu.classList.remove('active');
            }
        });
    }
}

// ========== UTILITY FUNCTIONS ==========
function toggleVoiceRecording() {
    if (isRecording) {
        stopVoiceRecording();
    } else {
        startVoiceRecording();
    }
}

function autoResize() {
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.style.height = 'auto';
        chatInput.style.height = Math.min(chatInput.scrollHeight, 150) + 'px';
    }
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showNotification('Message copied to clipboard', 'success');
        soundManager.playBeep("success");
    }).catch(() => {
        showNotification('Failed to copy message', 'error');
        soundManager.playBeep("error");
    });
}

function openFileModal() {
    const fileModal = document.getElementById('fileModal');
    if (fileModal) {
        fileModal.style.display = 'flex';
    }
}

function closeFileModal() {
    const fileModal = document.getElementById('fileModal');
    if (fileModal) {
        fileModal.style.display = 'none';
    }
    
    // Reset file input
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.value = '';
    }
    
    // Clear file preview
    const fileUploadArea = document.getElementById('fileUploadArea');
    if (fileUploadArea) {
        fileUploadArea.innerHTML = `
            <i class="fas fa-cloud-upload-alt"></i>
            <p>Drag & drop your file here or click to browse</p>
        `;
    }
}

function showFilePreview(file) {
    const fileUploadArea = document.getElementById('fileUploadArea');
    if (!fileUploadArea) return;
    
    fileUploadArea.innerHTML = `
        <div class="file-preview">
            <i class="fas fa-file"></i>
            <div class="file-info">
                <div class="file-name">${file.name}</div>
                <div class="file-size">${formatFileSize(file.size)}</div>
            </div>
        </div>
    `;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function showNotification(message, type = 'info') {
    // Create notification container if it doesn't exist
    let container = document.getElementById('notificationContainer');
    if (!container) {
        container = document.createElement('div');
        container.id = 'notificationContainer';
        container.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 10000;
            max-width: 400px;
        `;
        document.body.appendChild(container);
    }
    
    const notification = document.createElement('div');
    notification.className = 'notification';
    
    const colors = {
        success: '#27ae60',
        error: '#e74c3c',
        warning: '#f39c12',
        info: '#3498db'
    };
    
    notification.style.cssText = `
        background: rgba(0, 0, 0, 0.9);
        color: white;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-radius: 8px;
        border-left: 4px solid ${colors[type] || colors.info};
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        animation: slideIn 0.3s ease-out;
    `;
    
    notification.innerHTML = `
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : type === 'warning' ? 'exclamation-triangle' : 'info-circle'}" style="color: ${colors[type] || colors.info}"></i>
            <span>${message}</span>
        </div>
    `;
    
    container.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-in forwards';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 5000);
}

function showLoading(message = 'Processing...') {
    let loadingOverlay = document.getElementById('loadingOverlay');
    if (!loadingOverlay) {
        loadingOverlay = document.createElement('div');
        loadingOverlay.id = 'loadingOverlay';
        loadingOverlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
            backdrop-filter: blur(5px);
        `;
        
        loadingOverlay.innerHTML = `
            <div style="text-align: center; color: white;">
                <div style="width: 40px; height: 40px; border: 3px solid #333; border-top: 3px solid #00bcd4; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 1rem;"></div>
                <div id="loadingText">${message}</div>
            </div>
        `;
        
        document.body.appendChild(loadingOverlay);
    } else {
        loadingOverlay.style.display = 'flex';
        const loadingText = document.getElementById('loadingText');
        if (loadingText) loadingText.textContent = message;
    }
}

function hideLoading() {
    const loadingOverlay = document.getElementById('loadingOverlay');
    if (loadingOverlay) {
        loadingOverlay.style.display = 'none';
    }
}

function createStars() {
    const starsContainer = document.getElementById('stars-container');
    if (!starsContainer) return;
    
    // Create stars
    for (let i = 0; i < 100; i++) {
        const star = document.createElement('div');
        star.className = 'star';
        star.style.left = Math.random() * 100 + '%';
        star.style.top = Math.random() * 100 + '%';
        star.style.animationDuration = (Math.random() * 3 + 2) + 's';
        star.style.animationDelay = Math.random() * 2 + 's';
        starsContainer.appendChild(star);
    }
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .star {
        position: absolute;
        width: 2px;
        height: 2px;
        background: white;
        border-radius: 50%;
        opacity: 0;
        animation: twinkle 2s infinite;
    }
    
    @keyframes twinkle {
        0%, 100% { opacity: 0; }
        50% { opacity: 1; }
    }
    
    .typing-animation {
        display: flex;
        gap: 4px;
        padding: 10px;
    }
    
    .typing-animation span {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #666;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-animation span:nth-child(1) { animation-delay: 0s; }
    .typing-animation span:nth-child(2) { animation-delay: 0.2s; }
    .typing-animation span:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
        30% { transform: translateY(-10px); opacity: 1; }
    }
`;
document.head.appendChild(style);

// ========== INITIALIZATION COMPLETE ==========
console.log('✅ Complete Enhanced NOVA Frontend (1.8k+ lines) fully initialized with replit_backend.py integration');
console.log('🚨 FIXED: Specialized agents (emotional, medical) now provide AI responses for casual conversations!');
console.log('🔧 COMPLETE: Full feature integration matching original comprehensive system');