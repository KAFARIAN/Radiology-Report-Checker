// ====================================================================
// 1. GLOBAL VARIABLES AND CONFIG
// ====================================================================
const API_BASE_URL = 'http://localhost:5000'; 
const STORAGE_KEY = 'radiology_chat_history';
const MAX_SLOTS = 11;

// --- Core UI Elements ---
const input = document.getElementById('text-message');
const sendButton = document.getElementById('send-btn');
const chatbox = document.getElementById('chatbox');
const historyList = document.getElementById('history-list');
const newChatBtn = document.getElementById('new-chat');
const micButton = document.getElementById('mic-btn');
const imageUploadButton = document.getElementById('image-upload-btn');
const imageInput = document.getElementById('image-input');
const reportButton = document.getElementById('report-btn');
const extractButton = document.getElementById('extract-btn');

// --- State Variables ---
let currentSessionId = Date.now().toString();
let isNewSession = true;
let isSending = false;
let uploadedImageFile = null; 

// ====================================================================
// 2. HELPERS & RENDERING
// ====================================================================
function formatMessage(text) {
    return text
        .replace(/\*\*\*(.*?)\*\*\*/g, '<strong><em>$1</em></strong>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>')
        .replace(/^- (.*)$/gm, '<li>$1</li>');
}

function addMessageToChatbox(content, type) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', `${type}-message`);
    msgDiv.innerHTML = `<div class="content">${formatMessage(content)}</div>`;
    chatbox.appendChild(msgDiv);
    chatbox.scrollTo({ top: chatbox.scrollHeight, behavior: 'smooth' });
    return msgDiv;
}

function displayGeneratedImage(imageUrl) {
    // 1. Fix the URL concatenation to prevent "localhost:5000output/..."
    const cleanImagePath = imageUrl.startsWith('/') ? imageUrl.substring(1) : imageUrl;
    const fullUrl = imageUrl.startsWith('http') ? imageUrl : `${API_BASE_URL}/${cleanImagePath}`;
    
    const wrapper = document.createElement('div');
    wrapper.classList.add('message', 'bot-message', 'image-container');
    
    const imgElement = document.createElement('img');
    imgElement.src = fullUrl;
    imgElement.alt = "Medical Vis";
    imgElement.style.maxWidth = "100%";
    imgElement.style.borderRadius = "8px";
    imgElement.style.marginTop = "10px";
    imgElement.style.cursor = "pointer";

    // Add an error logger to see exactly what URL failed
    imgElement.onerror = () => console.error("Failed to load image from:", fullUrl);
    
    // 2. Scroll only AFTER the image is fully loaded
    imgElement.onload = () => {
        chatbox.scrollTo({ top: chatbox.scrollHeight, behavior: 'smooth' });
    };

    imgElement.onclick = () => window.open(fullUrl, '_blank');

    wrapper.appendChild(imgElement);
    chatbox.appendChild(wrapper);
}

function getAllSessions() {
    // This part is perfect, no changes needed.
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
}

// ====================================================================
// 3. UPDATED HISTORY & SESSION LOGIC
// ====================================================================
function handleSessionSaving(userQuestion, botReply) {
    const sessions = getAllSessions();
    
    if (isNewSession) {
        let cleanTitle = userQuestion.replace(/Please generate a professional.*?:\s*/i, "");
        cleanTitle = cleanTitle.replace(/Extract clinical data.*?:\s*/i, "");
        cleanTitle = cleanTitle.substring(0, 30) + (cleanTitle.length > 30 ? "..." : "");
        
        sessions[currentSessionId] = {
            id: currentSessionId,
            title: cleanTitle || "New Consultation",
            timestamp: Date.now(),
            html: "" 
        };
        isNewSession = false;
    }

    if (sessions[currentSessionId]) {
        sessions[currentSessionId].html = chatbox.innerHTML;
        sessions[currentSessionId].timestamp = Date.now(); 
    }

    const keys = Object.keys(sessions).sort((a,b) => sessions[b].timestamp - sessions[a].timestamp);
    if (keys.length > MAX_SLOTS) {
        for (let i = MAX_SLOTS; i < keys.length; i++) {
            delete sessions[keys[i]];
        }
    }

    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
    renderHistoryUI();
}

function renderHistoryUI() {
    const sessions = Object.values(getAllSessions()).sort((a, b) => b.timestamp - a.timestamp);
    historyList.innerHTML = '';

    if (sessions.length === 0) {
        historyList.innerHTML = '<p class="history" style="padding:15px; color:#888;">No History Yet...</p>';
        return;
    }

    sessions.forEach(session => {
        const item = document.createElement('div');
        item.className = `history-item ${session.id === currentSessionId ? 'active' : ''}`;
        item.innerHTML = `
            <div class="history-content" style="padding: 10px; cursor: pointer;">
                <span class="chat-title" style="color: white; display: block; overflow: hidden; white-space: nowrap; text-overflow: ellipsis;">
                    ${session.title}
                </span>
                <small style="color: #888; font-size: 0.8em;">
                    ${new Date(session.timestamp).toLocaleDateString()} ${new Date(session.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                </small>
            </div>
        `;

        item.onclick = () => {
            currentSessionId = session.id;
            isNewSession = false; 
            chatbox.innerHTML = session.html;
            renderHistoryUI(); 
            chatbox.scrollTop = chatbox.scrollHeight;
        };
        historyList.appendChild(item);
    });
}

// ====================================================================
// 4. CORE CHAT LOGIC (With Image Fixes)
// ====================================================================
imageUploadButton.addEventListener('click', () => imageInput.click());

imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        uploadedImageFile = file;
        const msg = addMessageToChatbox(`📎 Ready to upload: ${file.name}`, 'user');
        msg.style.opacity = "0.7"; 
    }
});

async function sendMessage(userInput = null, contextPrefix = "") {
    if (isSending) return;
    
    const rawText = userInput || input.value.trim();
    // Logic check: Allow sending if there's text OR an image file
    if (!rawText && !uploadedImageFile) return;

    isSending = true;
    let finalMessage = contextPrefix + rawText;
    
    if (rawText) addMessageToChatbox(rawText, 'user');
    input.value = '';
    
    const loadingDiv = addMessageToChatbox("Thinking...", 'bot');
    loadingDiv.classList.add('loading-indicator');

    try {
        // --- 1. Handling Image Upload ---
        if (uploadedImageFile) {
            loadingDiv.innerHTML = '<div class="content">Uploading medical scan...</div>';
            const formData = new FormData();
            formData.append('file', uploadedImageFile);

            const uploadResponse = await fetch(`${API_BASE_URL}/api/upload`, {
                method: 'POST',
                body: formData
            });

            if (!uploadResponse.ok) throw new Error("Image upload failed");
            const uploadData = await uploadResponse.json();
            
            // Note: Clear these BEFORE the next fetch to prevent accidental double-uploads
            uploadedImageFile = null;
            imageInput.value = ""; 
            
            finalMessage += `\n\n[SYSTEM: The user has uploaded an image for analysis at this path: ${uploadData.filepath}]`;
        }

        // --- 2. Requesting AI Response ---
        loadingDiv.innerHTML = '<div class="content">Analyzing findings...</div>';
        
        const response = await fetch(`${API_BASE_URL}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: finalMessage })
        });

        if (!response.ok) throw new Error(`Server error: ${response.status}`);

        const data = await response.json();
        
        // Safety check before removing
        if (loadingDiv && loadingDiv.parentNode) {
            chatbox.removeChild(loadingDiv);
        }

        let botReply = data.reply || "No response.";

        // --- 3. REFINED IMAGE DETECTION & CLEANUP ---
        // This regex specifically finds Markdown images ![]() OR raw URLs
        const markdownImageRegex = /!\[.*?\]\((https?:\/\/[^\s)]+\.(?:png|jpg|jpeg|gif))\)/i;
        const rawUrlRegex = /(https?:\/\/[^\s)]+\.(?:png|jpg|jpeg|gif))/i;

        let imageUrl = null;
        let match = botReply.match(markdownImageRegex);

        if (match) {
            imageUrl = match[1]; // Get the URL from inside the ![]()
            botReply = botReply.replace(match[0], '').trim(); // Remove the entire ![]() part
        } else {
            match = botReply.match(rawUrlRegex);
            if (match) {
                imageUrl = match[0];
                botReply = botReply.replace(imageUrl, '').trim(); // Remove raw URL
            }
        }

        // Display the text (if anything is left after removing the image link)
        if (botReply) {
            addMessageToChatbox(botReply, 'bot');
        }

        // Display the image if we found a URL
        if (imageUrl) {
            displayGeneratedImage(imageUrl);
        }

        handleSessionSaving(rawText, botReply);

    } catch (error) {
        if (loadingDiv && loadingDiv.parentNode) {
            chatbox.removeChild(loadingDiv);
        }
        console.error("Critical Error:", error);
        addMessageToChatbox(`Error: ${error.message}`, 'bot');
    } finally {
        isSending = false;
        // Small delay to ensure the UI is ready for focus
        setTimeout(() => input.focus(), 10);
    }
}

// ====================================================================
// 5. EVENT LISTENERS (Preventing Refresh)
// ====================================================================

newChatBtn.addEventListener('click', (e) => {
    e.preventDefault();
    currentSessionId = Date.now().toString();
    isNewSession = true;
    chatbox.innerHTML = '<div class="message bot-message"><div class="content">New session started. How can I help?</div></div>';
    renderHistoryUI();
});

sendButton.addEventListener('click', (e) => {
    e.preventDefault(); 
    sendMessage();
});

input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

reportButton.addEventListener('click', (e) => {
    e.preventDefault();
    if(!input.value.trim()) return addMessageToChatbox("Enter findings first.", 'bot');
    sendMessage(input.value.trim(), "Please generate a professional radiology report for: ");
    const patient = prompt("Patient info (e.g., 'John Doe, Male, 45 years old'):");
    const findings = input.value;
    sendMessage(`${patient} ${findings}`, "Generate full radiology report for ");
});

extractButton.addEventListener('click', (e) => {
    e.preventDefault();
    if(!input.value.trim()) return addMessageToChatbox("Enter text first.", 'bot');
    sendMessage(input.value.trim(), "Extract clinical data to JSON from: ");
});

document.addEventListener('DOMContentLoaded', renderHistoryUI);

// Speech Recognition
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (SpeechRecognition) {
    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.continuous = false;
    recognition.onresult = (event) => {
        input.value = event.results[0][0].transcript;
        micButton.classList.remove('active');
    };
    recognition.onend = () => micButton.classList.remove('active');
    micButton.addEventListener('click', () => {
        if (micButton.classList.contains('active')) recognition.stop();
        else {
            micButton.classList.add('active');
            recognition.start();
        }
    });
} else {
    micButton.style.display = 'none';
}
