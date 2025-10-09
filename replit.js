// ========== NOVA ULTRA PROFESSIONAL FRONTEND - REPLIT BACKEND INTEGRATION ==========
// Full integration with replit_backend.py FastAPI endpoints
// Version: 3.0.0-free-apis-enhanced

// BACKEND CONFIGURATION - EXACT MATCH WITH replit_backend.py
const API_BASE = "http://127.0.0.1:8000"; // Match your backend port (8000 from uvicorn.run)

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

// COMPLETE FORM SUBMISSION PREVENTION
document.addEventListener('DOMContentLoaded', function () {
    // Prevent ALL form submissions globally
    document.addEventListener('submit', function (e) {
        e.preventDefault();
    });

    // Handle Enter key inside chatInput
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.addEventListener('keydown', function (e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                document.getElementById('sendBtn')?.click();
            }
        });
    }

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

// ========== BACKEND API INTEGRATION - ALL ENDPOINTS FROM replit_backend.py ==========

// EXACT BACKEND INTEGRATION: GET / endpoint (Root)
async function getSystemInfo() {
    try {
        console.log('📡 Calling GET / endpoint...');
        const response = await fetch(`${API_BASE}/`, {
            method: 'GET'
        });
        
        if (response.ok) {
            const data = await response.json();
            console.log('✅ System Info Response:', data);
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

    // Sound feedback
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
        console.log('📡 Calling POST /chat endpoint...');
        
        // EXACT backend POST /chat endpoint with ChatRequest model
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                user_id: userId
            })
        });

        if (!response.ok) {
            throw new Error(`Backend error: ${response.status} - ${response.statusText}`);
        }

        const data = await response.json();
        removeTypingIndicator(typingId);
        
        console.log('✅ Chat Response:', data);
        
        // Process ChatResponse model response - ALL FIELDS from backend
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
        const mlEnhanced = data.ml_enhanced || false;
        const contextUsed = data.context_used || false;
        const recommendations = data.recommendations || [];
        const enhancementReason = data.enhancement_reason || '';
        
        // Add message with full metadata
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
            contextUsed: contextUsed,
            recommendations: recommendations,
            enhancementReason: enhancementReason
        });

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

        // Show ML enhancement info if available
        if (mlEnhanced) {
            showNotification(`ML Enhanced Response: ${enhancementReason}`, 'info');
        }

    } catch (err) {
        console.error('❌ Chat error:', err);
        removeTypingIndicator(typingId);
        addMessageToChat('bot', `❌ Connection Error: ${err.message}. Please ensure the NOVA backend is running on ${API_BASE}`);
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
        console.log('📡 Calling POST /file/upload endpoint...');
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('user_id', userId);
        formData.append('prompt', prompt || 'analyze this file');
        formData.append('analysis_type', 'comprehensive');
        
        const response = await fetch(`${API_BASE}/file/upload`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        removeTypingIndicator(typingId);
        
        console.log('✅ File Upload Response:', data);

        if (data.success && data.file && data.analysis) {
            const fileInfo = data.file;
            const analysis = data.analysis;
            const metadata = data.metadata || {};
            const performance = data.performance || {};
            
            let responseText = `📄 **File Uploaded Successfully!**\n\n`;
            
            // File Information
            responseText += `**File:** ${fileInfo.filename}\n`;
            responseText += `**Type:** ${fileInfo.type}\n`;
            responseText += `**Size:** ${fileInfo.size_kb} KB`;
            if (fileInfo.size_mb > 1) {
                responseText += ` (${fileInfo.size_mb} MB)`;
            }
            responseText += `\n`;
            
            // Processing Information
            if (analysis.strategy_used) {
                const strategyNames = {
                    'full_content': 'Full Content Analysis',
                    'intelligent_chunking': 'Smart Chunking + Semantic Search',
                    'summarization_pipeline': 'Progressive Summarization'
                };
                responseText += `**Strategy:** ${strategyNames[analysis.strategy_used] || analysis.strategy_used}\n`;
            }
            
            if (analysis.chunks_processed) {
                responseText += `**Sections Analyzed:** ${analysis.chunks_processed}\n`;
            }
            
            if (analysis.processing_time) {
                responseText += `**Processing Time:** ${analysis.processing_time}s\n`;
            }
            
            // AI Provider Information
            if (metadata.ai_metadata) {
                const aiMeta = metadata.ai_metadata;
                responseText += `**AI Provider:** ${aiMeta.api_provider || 'Unknown'}\n`;
                if (aiMeta.ml_enhanced) {
                    responseText += `**ML Enhanced:** Yes 🧠\n`;
                }
            }
            
            // Performance Score
            if (performance.efficiency_score) {
                responseText += `**Efficiency Score:** ${performance.efficiency_score}/10\n`;
            }
            
            responseText += `\n🤖 **AI Analysis:**\n\n${analysis.response}`;
            
            // Add processing details if available
            if (metadata.processing_strategy) {
                responseText += `\n\n📊 **Processing Details:**\n`;
                responseText += `• Strategy: ${metadata.processing_strategy.reason}\n`;
                if (metadata.processing_strategy.embedding_enhanced) {
                    responseText += `• Embedding Enhanced: Yes\n`;
                }
            }
            
            addMessageToChat('bot', responseText);
            soundManager.playBeep("success");
            
            // Save enhanced chat history
            saveChatToHistory(
                `[File: ${fileInfo.filename}] ${displayMessage}`,
                analysis.response
            );
            
            const strategyName = {
                'full_content': 'Full Content Analysis',
                'intelligent_chunking': 'Smart Chunking + Semantic Search',
                'summarization_pipeline': 'Progressive Summarization'
            }[analysis.strategy_used] || 'Advanced Processing';
            
            showNotification(`File processed successfully using ${strategyName}! 📄`, 'success');
            
        } else {
            // Enhanced error handling
            let errorMessage = data.message || 'File processing failed';
            
            // Show specific error details if available
            if (data.error_details) {
                console.error('Detailed error:', data.error_details);
            }
            
            // Show fallback response if provided
            if (data.fallback_response) {
                addMessageToChat('bot', `❌ ${errorMessage}\n\n📋 **Guidance:**\n${data.fallback_response}`);
            } else {
                addMessageToChat('bot', `❌ File processing failed: ${errorMessage}`);
            }
            
            // Show troubleshooting info if available
            if (data.support_info) {
                const support = data.support_info;
                let supportText = '\n\n🔧 **Troubleshooting:**\n';
                supportText += `• Max file size: ${support.max_file_size}\n`;
                supportText += `• Supported formats: ${support.supported_formats.join(', ')}\n`;
                if (support.troubleshooting) {
                    supportText += `• Steps to try:\n`;
                    support.troubleshooting.forEach(step => {
                        supportText += `  - ${step}\n`;
                    });
                }
                addMessageToChat('bot', supportText);
            }
            
            soundManager.playBeep("error");
            showNotification(`File processing failed: ${errorMessage}`, 'error');
        }
        
    } catch (error) {
        console.error('❌ File upload error:', error);
        removeTypingIndicator(typingId);
        
        let errorResponse = `❌ File processing failed: ${error.message}`;
        
        // Handle specific error cases
        if (error.message.includes('413') || error.message.includes('File too large')) {
            errorResponse += '\n\n📏 **File too large!** Please try a file smaller than 50MB.';
        } else if (error.message.includes('422') || error.message.includes('Unprocessable')) {
            errorResponse += '\n\n🔧 **File format issue!** Please ensure your file contains readable content.';
        } else if (error.message.includes('429') || error.message.includes('Rate limit')) {
            errorResponse += '\n\n⏳ **Rate limit reached!** Please wait before uploading another file.';
        } else if (error.message.includes('500')) {
            errorResponse += '\n\n🔄 **Server error!** Please try again in a moment.';
        }
        
        addMessageToChat('bot', errorResponse);
        soundManager.playBeep("error");
        showNotification('File processing failed: ' + error.message, 'error');
    }
}

// File size formatting function
function formatFileSize(bytes) {
    if (!bytes || bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Enhanced file validation
function validateFileForUpload(file) {
    const maxSize = 50 * 1024 * 1024; // 50MB
    const supportedExtensions = [
        '.pdf', '.docx', '.doc', '.txt', '.csv', '.xlsx', '.xls', 
        '.json', '.js', '.html', '.css', '.py', '.md', '.sql'
    ];
    
    if (file.size > maxSize) {
        return {
            valid: false,
            error: `File too large (${formatFileSize(file.size)}). Maximum size is 50MB.`
        };
    }
    
    const extension = '.' + file.name.split('.').pop().toLowerCase();
    const isTypeSupported = supportedExtensions.includes(extension);
    
    if (!isTypeSupported) {
        return {
            valid: false,
            error: `Unsupported file type. Supported formats: PDF, DOCX, TXT, CSV, XLSX, JSON, Code files.`
        };
    }
    
    return { valid: true };
}

// File upload with validation
async function handleFileUpload() {
    const fileInput = document.getElementById('fileInput');
    if (!fileInput || !fileInput.files[0]) return;
    
    const file = fileInput.files[0];
    const validation = validateFileForUpload(file);
    
    if (!validation.valid) {
        showNotification(validation.error, 'error');
        soundManager.playBeep("error");
        return;
    }
    
    // Show file info before processing
    const fileInfo = `📎 Selected: ${file.name} (${formatFileSize(file.size)})`;
    console.log(fileInfo);
    
    // Proceed with upload
    await processFileUpload();
}

// Add drag and drop support
function setupDragAndDrop() {
    const chatContainer = document.querySelector('.chat-container');
    if (!chatContainer) return;
    
    chatContainer.addEventListener('dragover', (e) => {
        e.preventDefault();
        chatContainer.classList.add('drag-over');
    });
    
    chatContainer.addEventListener('dragleave', (e) => {
        e.preventDefault();
        chatContainer.classList.remove('drag-over');
    });
    
    chatContainer.addEventListener('drop', (e) => {
        e.preventDefault();
        chatContainer.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const fileInput = document.getElementById('fileInput');
            if (fileInput) {
                fileInput.files = files;
                handleFileUpload();
            }
        }
    });
}

// Initialize enhanced features
document.addEventListener('DOMContentLoaded', function() {
    setupDragAndDrop();
    
    // Add enhanced file input change listener
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.addEventListener('change', handleFileUpload);
    }
});

// EXACT BACKEND INTEGRATION: POST /voice/process endpoint
async function processVoiceCommand(audioBlob = null, text = null) {
    try {
        console.log('📡 Calling POST /voice/process endpoint...');
        
        const formData = new FormData();
        if (audioBlob) {
            formData.append('audio', audioBlob, 'voice-input.webm');
        }
        if (text) {
            formData.append('text', text);
        }
        formData.append('user_id', userId);

        const response = await fetch(`${API_BASE}/voice/process`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Voice processing error: ${response.status}`);
        }
        
        // Get audio blob and play it
        const audioData = await response.blob();
        const audioUrl = URL.createObjectURL(audioData);
        const audio = new Audio(audioUrl);
        
        audio.onloadeddata = () => {
            showNotification('🔊 Playing AI voice response...', 'success');
            soundManager.playBeep("success");
        };
        
        audio.onended = () => {
            URL.revokeObjectURL(audioUrl);
            showNotification('🔊 Voice response complete', 'success');
            soundManager.playBeep("success");
        };
        
        audio.onerror = () => {
            showNotification('❌ Audio playback failed', 'error');
            soundManager.playBeep("error");
        };
        
        await audio.play();
        return true;
        
    } catch (error) {
        console.error('❌ Voice processing error:', error);
        showNotification('Voice processing failed: ' + error.message, 'error');
        soundManager.playBeep("error");
        return false;
    }
}

// Voice recording functions
async function startVoiceRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) {
                audioChunks.push(e.data);
            }
        };
        
        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            await processVoiceCommand(audioBlob);
            stream.getTracks().forEach(track => track.stop());
        };
        
        mediaRecorder.start(100);
        isRecording = true;
        updateVoiceButtonState();
        showNotification('Recording... Speak now', 'info');
        soundManager.playBeep("click");
        
        // Auto-stop after 10 seconds
        setTimeout(() => {
            if (isRecording) stopVoiceRecording();
        }, 10000);
        
    } catch (err) {
        console.error('Microphone error:', err);
        showNotification('Could not access microphone', 'error');
        soundManager.playBeep("error");
    }
}

function stopVoiceRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        updateVoiceButtonState();
        soundManager.playBeep("click");
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

// Text-to-Speech function
async function speakText(text) {
    if (!text) return false;
    
    try {
        showNotification('🔊 Generating speech...', 'info');
        return await processVoiceCommand(null, text);
    } catch (error) {
        console.error('TTS error:', error);
        showNotification('TTS error: ' + error.message, 'error');
        soundManager.playBeep("error");
        return false;
    }
}

// EXACT BACKEND INTEGRATION: POST /github/analyze endpoint
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
        console.log('📡 Calling POST /github/analyze endpoint...');
        
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
        console.log('✅ GitHub Analysis Response:', data);

        if (data.success) {
            const repoAnalysisEl = document.getElementById('repoAnalysis');
            const codeQualityEl = document.getElementById('codeQuality');
            const debugSuggestionsEl = document.getElementById('debugSuggestions');

            const repoUrlMeta = data.metadata?.repo_url || repoUrl;
            const mlEnhanced = data.metadata?.ml_enhanced || false;
            const processingTime = data.metadata?.processing_time || 0;

            const analysisText = `📂 **Repository Analysis Complete**

**Repository:** ${repoUrlMeta}
**ML Enhanced:** ${mlEnhanced ? 'Yes' : 'No'}
**Processing Time:** ${processingTime.toFixed(2)}s
**Agent Used:** ${data.metadata?.agent_used || 'coding'}

---

🤖 **AI Analysis:**
${data.response || 'No AI response generated'}

---

**Analysis Metadata:**
${JSON.stringify(data.metadata, null, 2)}
`;

            if (repoAnalysisEl) {
                repoAnalysisEl.textContent = analysisText;
            }
            if (codeQualityEl) {
                codeQualityEl.textContent = 'Analysis completed - see detailed report above';
            }
            if (debugSuggestionsEl) {
                debugSuggestionsEl.textContent = 'Recommendations provided in the analysis report';
            }

            showNotification('Repository analyzed successfully! 🎉', 'success');
            soundManager.playBeep("success");

        } else {
            throw new Error(data.message || 'GitHub analysis failed');
        }

    } catch (error) {
        hideLoading();
        console.error('❌ GitHub analysis error:', error);
        showNotification('GitHub analysis failed: ' + error.message, 'error');
        soundManager.playBeep("error");

        const repoAnalysisEl = document.getElementById('repoAnalysis');
        if (repoAnalysisEl) {
            repoAnalysisEl.textContent = 'Analysis failed: ' + error.message;
        }
    }
}

// EXACT BACKEND INTEGRATION: POST /github/question endpoint
async function askGitHubQuestion(question) {
    if (!question.trim()) return;

    soundManager.playBeep("click");

    try {
        console.log('📡 Calling POST /github/question endpoint...');
        
        const formData = new FormData();
        formData.append('question', question);
        formData.append('user_id', userId || 'web-user');

        const response = await fetch(`${API_BASE}/github/question`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error(`GitHub question error: ${response.status}`);
        const data = await response.json();

        console.log('✅ GitHub Question Response:', data);

        if (data.success) {
            soundManager.playBeep("success");
            return data.response;
        } else {
            throw new Error(data.message || 'GitHub question failed');
        }

    } catch (error) {
        console.error('❌ GitHub question error:', error);
        soundManager.playBeep("error");
        return `Error asking GitHub question: ${error.message}`;
    }
}

// EXACT BACKEND INTEGRATION: GET /agents endpoint
async function getAvailableAgents() {
    try {
        console.log('📡 Calling GET /agents endpoint...');
        const response = await fetch(`${API_BASE}/agents`, {
            method: 'GET'
        });
        
        if (response.ok) {
            const data = await response.json();
            console.log('✅ Available Agents:', data);
            return data;
        } else {
            throw new Error(`Get agents failed: ${response.status}`);
        }
    } catch (error) {
        console.error('❌ Get agents error:', error);
        return null;
    }
}

// EXACT BACKEND INTEGRATION: GET /system endpoint
async function getSystemStatus() {
    try {
        console.log('📡 Calling GET /system endpoint...');
        const response = await fetch(`${API_BASE}/system`, {
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
        console.log('📡 Calling POST /clear/{user_id} endpoint...');
        const response = await fetch(`${API_BASE}/clear/${userId}`, {
            method: 'POST'
        });
        
        if (response.ok) {
            const data = await response.json();
            console.log('✅ Clear Context Response:', data);
            
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
        console.log('📡 Calling GET /health endpoint...');
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
    
    if (command === '/clear' || command === '/reset') {
        await clearContext();
        return true;
    }
    
    if (command === '/status' || command === '/health') {
        const status = await getSystemStatus();
        if (status) {
            let statusText = '📊 **System Status:**\n\n';
            
            statusText += `**Status:** ${status.status}\n`;
            statusText += `**Version:** ${status.version}\n`;
            statusText += `**Timestamp:** ${status.timestamp}\n\n`;
            
            statusText += '**Components:**\n';
            for (const [key, value] of Object.entries(status.components || {})) {
                statusText += `• ${key}: ${value}\n`;
            }
            
            statusText += '\n**Capabilities:**\n';
            for (const [key, value] of Object.entries(status.capabilities || {})) {
                statusText += `• ${key}: ${value}\n`;
            }
            
            if (status.api_providers) {
                statusText += '\n**API Providers:**\n';
                statusText += `• Available: ${status.api_providers.available_providers}\n`;
                statusText += `• Current: ${status.api_providers.current_provider || 'None'}\n`;
            }
            
            addMessageToChat('bot', statusText);
        } else {
            addMessageToChat('bot', '❌ Unable to retrieve system status');
        }
        return true;
    }
    
    if (command === '/agents') {
        const agents = await getAvailableAgents();
        if (agents) {
            let agentsText = '🤖 **Available Agents:**\n\n';
            
            for (const [agentType, agentInfo] of Object.entries(agents.agents || {})) {
                agentsText += `**${agentInfo.emoji || '🤖'} ${agentInfo.name}**\n`;
                agentsText += `${agentInfo.description}\n`;
                agentsText += `Specialties: ${agentInfo.specialties?.join(', ') || 'General'}\n`;
                agentsText += `ML Enhanced: ${agentInfo.ml_enhanced ? 'Yes' : 'No'}\n\n`;
            }
            
            agentsText += `**System Info:**\n`;
            agentsText += `• ML System: ${agents.ml_system_available ? 'Available' : 'Basic'}\n`;
            agentsText += `• Smart Enhancement: ${agents.smart_enhancement ? 'Yes' : 'No'}\n`;
            agentsText += `• Always AI Response: ${agents.always_ai_response ? 'Yes' : 'No'}\n`;
            
            addMessageToChat('bot', agentsText);
        } else {
            addMessageToChat('bot', '❌ Unable to retrieve agents information');
        }
        return true;
    }
    
    if (command === '/help' || command === '/commands') {
        const helpText = `🤖 **Available Commands:**

**Chat Commands:**
• Just type normally to chat with NOVA
• Use the voice button 🎤 for speech input/output
• Use the paperclip 📎 to upload files

**System Commands:**
• \`/clear\` - Clear conversation context
• \`/reset\` - Same as clear
• \`/status\` - Show system status
• \`/health\` - Show backend health
• \`/agents\` - Show available agents
• \`/help\` - Show this help message

**Special Features:**
• File upload and AI analysis
• GitHub repository analysis
• Multi-agent AI system with ML routing
• Voice input/output processing
• Memory persistence with context
• Smart enhancement detection

**Backend Integration:**
• All endpoints from replit_backend.py
• FastAPI ChatRequest/ChatResponse models
• Full ML enhancement pipeline
• Free API providers (Groq, OpenRouter, etc.)

Just start typing to begin chatting with NOVA! 🚀`;
        
        addMessageToChat('bot', helpText);
        return true;
    }
    
    if (command.startsWith('/github ')) {
        const repoUrl = message.substring(8).trim();
        if (isValidGitHubUrl(repoUrl)) {
            addMessageToChat('user', `🔍 Analyzing GitHub repository: ${repoUrl}`);
            const repoUrlInput = document.getElementById('githubRepoUrl');
            if (repoUrlInput) {
                repoUrlInput.value = repoUrl;
                await analyzeGitHubRepo();
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
    if (chatMessages) {
        chatMessages.innerHTML = `
            <div class="message bot">
                <div class="font-medium text-cyan-400 mb-1">NOVA</div>
                <div>Hello! I'm NOVA, your AI assistant. Let's start a new conversation. 🚀</div>
            </div>
        `;
    }

    // Reset input
    if (chatInput) {
        chatInput.value = '';
    }

    // Close floating menu
    const floatingMenu = document.getElementById('floatingMenu');
    if (floatingMenu) {
        floatingMenu.classList.remove('active');
    }
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
            // Silently fail if audio context is not available
        }
    }
}

// Global sound manager instance
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
        newChatMenuItem: () => startNewChat(),
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

    // Sound effects for menu items
    document.querySelectorAll('.menu-item').forEach(item => {
        item.addEventListener('click', () => {
            soundManager.playBeep("click");
        });
    });
}

// Chat Input and Send Button Event Listeners
function setupChatEventListeners() {
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');
    const voiceBtn = document.getElementById('voiceBtn');

    if (chatInput) {
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
        sendBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            sendMessage();
            return false;
        });
    }
    
    if (voiceBtn) {
        voiceBtn.addEventListener('click', async (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (isRecording) {
                stopVoiceRecording();
            } else {
                await startVoiceRecording();
            }
            return false;
        });
    }
}

function autoResize() {
    const textarea = document.getElementById('chatInput');
    if (textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
    }
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
        
        // Enhanced metadata display with ML enhancement info
        if (metadata.language || metadata.emotion || metadata.mlEnhanced) {
            metadataInfo = `<div class="text-xs text-gray-500 mt-1">`;
            if (metadata.language) metadataInfo += `Language: ${metadata.language}`;
            if (metadata.emotion) metadataInfo += ` | Emotion: ${metadata.emotion}`;
            if (metadata.conversationCount) metadataInfo += ` | Count: ${metadata.conversationCount}`;
            if (metadata.fileContextUsed) metadataInfo += ` | File Context: 📎`;
            if (metadata.mlEnhanced) metadataInfo += ` | ML Enhanced: ✨`;
            if (metadata.contextUsed) metadataInfo += ` | Context Used: 📚`;
            metadataInfo += `</div>`;
        }
        
        // Show enhancement reason if available
        if (metadata.enhancementReason) {
            metadataInfo += `<div class="text-xs text-blue-400 mt-1">💡 ${metadata.enhancementReason}</div>`;
        }
        
        // Show recommendations if available
        if (metadata.recommendations && metadata.recommendations.length > 0) {
            metadataInfo += `<div class="text-xs text-green-400 mt-1">🎯 Recommendations: ${metadata.recommendations.join(', ')}</div>`;
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
    console.log(`Attempting to navigate to page ${pageNumber}`);
    
    // Hide all pages
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
        page.style.display = 'none'; // Force hide
    });
    
    // Show target page
    const targetPage = document.getElementById(`page${pageNumber}`);
    if (targetPage) {
        targetPage.style.display = 'block'; // Force show FIRST
        // Small delay to ensure display is applied before transition
        requestAnimationFrame(() => {
            targetPage.classList.add('active');
        });
        currentPage = pageNumber;
        console.log(`Successfully navigated to page ${pageNumber}`);
    } else {
        console.error(`Page ${pageNumber} not found in DOM`);
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
    
    // Scroll to top
    window.scrollTo(0, 0);
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
            console.log(`🤖 Backend Version: ${systemInfo.version || 'Unknown'}`);
            console.log(`📊 Status: ${systemInfo.status || 'Unknown'}`);
            console.log(`🎯 Features: ${systemInfo.features?.length || 0} available`);
            
            showNotification('NOVA Backend Connected! 🚀', 'success');
            
            if (status) {
                console.log('📈 System Status Loaded');
                console.log(`💥 Components: ${Object.keys(status.components || {}).length}`);
                console.log(`🔗 API Providers: ${status.api_providers?.available_providers || 0}`);
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
    if (event.error && event.error.message && event.error.message.includes('fetch')) {
        showNotification('Connection error. Please check if the backend server is running.', 'error');
    }
});

window.addEventListener('unhandledrejection', function(event) {
    console.error('🚨 Unhandled Promise Rejection:', event.reason);
    if (event.reason && event.reason.message && event.reason.message.includes('fetch')) {
        showNotification('API request failed. Please check backend connection.', 'error');
    }
});

// ========== ADVANCED FEATURES ==========

// GitHub repository question handler for chat
function handleGitHubQuestionInChat(question) {
    addMessageToChat('user', `🔍 GitHub Question: ${question}`);
    const typingId = addTypingIndicator();
    
    askGitHubQuestion(question).then(answer => {
        removeTypingIndicator(typingId);
        addMessageToChat('bot', `**GitHub Repository Answer:**\n\n${answer}`);
        saveChatToHistory(`GitHub Question: ${question}`, answer);
    }).catch(error => {
        removeTypingIndicator(typingId);
        addMessageToChat('bot', `❌ GitHub question failed: ${error.message}`);
    });
}

// Enhanced system diagnostics
async function runSystemDiagnostics() {
    showLoading('Running system diagnostics...');
    
    try {
        const diagnostics = {
            backend_connection: false,
            endpoints_available: 0,
            response_times: {},
            features_working: {},
            errors: []
        };
        
        // Test each endpoint
        const endpointTests = [
            { name: 'root', test: () => getSystemInfo() },
            { name: 'health', test: () => healthCheck() },
            { name: 'system', test: () => getSystemStatus() },
            { name: 'agents', test: () => getAvailableAgents() }
        ];
        
        for (const endpoint of endpointTests) {
            try {
                const startTime = Date.now();
                const result = await endpoint.test();
                const endTime = Date.now();
                
                diagnostics.response_times[endpoint.name] = endTime - startTime;
                diagnostics.features_working[endpoint.name] = !!result;
                if (result) diagnostics.endpoints_available++;
            } catch (error) {
                diagnostics.features_working[endpoint.name] = false;
                diagnostics.errors.push(`${endpoint.name}: ${error.message}`);
            }
        }
        
        diagnostics.backend_connection = diagnostics.endpoints_available > 0;
        
        hideLoading();
        
        let diagnosticsText = '🔧 **System Diagnostics Report**\n\n';
        diagnosticsText += `**Backend Connection:** ${diagnostics.backend_connection ? '✅ Connected' : '❌ Failed'}\n`;
        diagnosticsText += `**Endpoints Available:** ${diagnostics.endpoints_available}/4\n\n`;
        
        diagnosticsText += '**Response Times:**\n';
        for (const [endpoint, time] of Object.entries(diagnostics.response_times)) {
            diagnosticsText += `• ${endpoint}: ${time}ms\n`;
        }
        
        diagnosticsText += '\n**Feature Status:**\n';
        for (const [feature, working] of Object.entries(diagnostics.features_working)) {
            diagnosticsText += `• ${feature}: ${working ? '✅ Working' : '❌ Failed'}\n`;
        }
        
        if (diagnostics.errors.length > 0) {
            diagnosticsText += '\n**Errors:**\n';
            diagnostics.errors.forEach(error => {
                diagnosticsText += `• ${error}\n`;
            });
        }
        
        addMessageToChat('bot', diagnosticsText);
        
    } catch (error) {
        hideLoading();
        addMessageToChat('bot', `❌ Diagnostics failed: ${error.message}`);
    }
}

// Batch file upload support
function handleMultipleFiles(files) {
    if (!files || files.length === 0) return;
    
    showNotification(`Processing ${files.length} files...`, 'info');
    
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        setTimeout(() => {
            // Simulate file selection for each file
            const fileInput = document.getElementById('fileInput');
            if (fileInput) {
                // Create a new FileList with single file
                const dt = new DataTransfer();
                dt.items.add(file);
                fileInput.files = dt.files;
                
                // Trigger file processing
                handleFileSelect({ target: { files: [file] } });
            }
        }, i * 1000); // Process files with 1 second delay
    }
}

// Export chat history
function exportChatHistory() {
    try {
        const history = loadChatHistory();
        if (history.length === 0) {
            showNotification('No chat history to export', 'warning');
            return;
        }
        
        const exportData = {
            exported_at: new Date().toISOString(),
            user_id: userId,
            total_conversations: history.length,
            conversations: history.map(chat => ({
                timestamp: chat.timestamp,
                agent: chat.agent,
                user_message: chat.userMessage,
                bot_response: chat.botResponse,
                session: chat.session
            }))
        };
        
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `nova-chat-history-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        showNotification(`Exported ${history.length} conversations`, 'success');
        
    } catch (error) {
        showNotification('Export failed: ' + error.message, 'error');
    }
}

// Import chat history
function importChatHistory(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const importData = JSON.parse(e.target.result);
            
            if (importData.conversations && Array.isArray(importData.conversations)) {
                // Merge with existing history
                const existingHistory = loadChatHistory();
                const mergedHistory = [...importData.conversations.map(conv => ({
                    id: Date.now() + Math.random(),
                    timestamp: conv.timestamp,
                    userMessage: conv.user_message,
                    botResponse: conv.bot_response,
                    agent: conv.agent || 'general',
                    session: conv.session || 'imported'
                })), ...existingHistory];
                
                // Keep only last 100 conversations
                chatHistory = mergedHistory.slice(0, 100);
                localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
                
                showNotification(`Imported ${importData.conversations.length} conversations`, 'success');
            } else {
                throw new Error('Invalid chat history format');
            }
        } catch (error) {
            showNotification('Import failed: ' + error.message, 'error');
        }
    };
    reader.readAsText(file);
}

// ========== FINAL INITIALIZATION ==========
window.addEventListener('load', function() {
    console.log('🚀 NOVA Ultra Professional Frontend Initialized');
    console.log('🔐 Authentication: Demo Mode (user@gmail.com / password123)');
    console.log('🎯 Backend API:', API_BASE);
    console.log('💡 All replit_backend.py endpoints integrated with exact FastAPI logic matching');
    console.log('👤 User ID:', userId);
    console.log('📡 Available Endpoints:');
    console.log('   - GET / (System Info)');
    console.log('   - POST /chat (Chat with AI)');
    console.log('   - POST /file/upload (File Analysis)');
    console.log('   - POST /voice/process (Voice Processing)');
    console.log('   - POST /github/analyze (GitHub Analysis)');
    console.log('   - POST /github/question (GitHub Q&A)');
    console.log('   - GET /agents (Available Agents)');
    console.log('   - GET /system (System Status)');
    console.log('   - GET /health (Health Check)');
    console.log('   - POST /clear/{user_id} (Clear Context)');
    
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
    
    // Initialize system check with delay
    setTimeout(() => {
        initializeSystemCheck();
    }, 1000);
    
    // Add development helper functions
    window.NOVA = {
        // Core API functions (exact from backend)
        sendMessage,
        clearContext,
        getSystemStatus,
        getSystemInfo,
        healthCheck,
        getAvailableAgents,
        processFileUpload,
        analyzeGitHubRepo,
        askGitHubQuestion,
        speakText,
        processVoiceCommand,
        
        // Special features
        handleSpecialCommands,
        runSystemDiagnostics,
        exportChatHistory,
        importChatHistory,
        
        // Navigation and UI
        navigateToPage,
        showNotification,
        addMessageToChat,
        toggleVoiceMode,
        startNewChat,
        
        // State and config
        API_BASE,
        userId,
        currentSession,
        selectedAgent,
        chatHistory,
        voiceModeEnabled,
        isProcessing,
        
        // Utility functions
        formatFileSize,
        isValidGitHubUrl,
        generateSessionId,
        getAgentInfo,
        
        // Sound system
        soundManager,
        
        // Advanced features
        handleMultipleFiles,
        handleGitHubQuestionInChat
    };
    
    console.log('🎉 NOVA Frontend Ready! Access helper functions via window.NOVA');
    console.log('📚 Available NOVA functions:', Object.keys(window.NOVA));
    console.log('🔧 Backend Integration Status: Complete with replit_backend.py');
    console.log('✨ Features: Always AI Response, ML Enhancement Detection, Free API Providers');
    
    // Add enhanced command suggestions
    const commandSuggestions = [
        'Type normally to chat with AI',
        'Use /help for available commands',
        'Upload files with the 📎 button',
        'Try /status to check system health',
        'Use /clear to reset conversation',
        'Analyze GitHub repos with /github [url]',
        'Enable voice mode for audio responses'
    ];
    
    console.log('💡 Quick Tips:');
    commandSuggestions.forEach(tip => console.log(`   • ${tip}`));
});

window.addEventListener('load', function() {
    const testDiv = document.createElement('div');
    testDiv.className = 'flex';
    document.body.appendChild(testDiv);
    const isFlexbox = window.getComputedStyle(testDiv).display === 'flex';
    document.body.removeChild(testDiv);
    
    if (!isFlexbox) {
        console.warn('Tailwind CSS may not be loaded properly');
    } else {
        console.log('Tailwind CSS verified');
    }
});