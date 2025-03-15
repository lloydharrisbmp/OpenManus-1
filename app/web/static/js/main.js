/**
 * OpenManus Financial Adviser Suite - Main JavaScript
 */

// Utility functions
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString(undefined, { 
        month: 'short', 
        day: 'numeric',
        year: date.getFullYear() !== new Date().getFullYear() ? 'numeric' : undefined
    });
}

function showToast(message, type = 'info') {
    // Create toast container if it doesn't exist
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container';
        document.body.appendChild(toastContainer);
    }
    
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <div class="toast-content">
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
            <span>${message}</span>
        </div>
    `;
    
    // Add to container
    toastContainer.appendChild(toast);
    
    // Show toast
    setTimeout(() => {
        toast.classList.add('show');
    }, 100);
    
    // Remove after 3 seconds
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => {
            toast.remove();
        }, 300);
    }, 3000);
}

// Mobile UI helpers
function toggleSidebar(sidebarSelector) {
    const sidebar = document.querySelector(sidebarSelector);
    if (sidebar.classList.contains('visible')) {
        sidebar.classList.remove('visible');
    } else {
        sidebar.classList.add('visible');
    }
}

// Add mobile controls if screen width is below threshold
function setupMobileControls() {
    const isMobile = window.innerWidth < 992;
    
    // Add mobile toggle buttons if they don't exist
    if (isMobile) {
        // Left sidebar toggle
        if (!document.getElementById('leftSidebarToggle')) {
            const leftToggle = document.createElement('button');
            leftToggle.id = 'leftSidebarToggle';
            leftToggle.className = 'mobile-sidebar-toggle left';
            leftToggle.innerHTML = '<i class="fas fa-bars"></i>';
            leftToggle.addEventListener('click', () => {
                toggleSidebar('.conversations-sidebar');
            });
            document.body.appendChild(leftToggle);
        }
        
        // Right sidebar toggle
        if (!document.getElementById('rightSidebarToggle')) {
            const rightToggle = document.createElement('button');
            rightToggle.id = 'rightSidebarToggle';
            rightToggle.className = 'mobile-sidebar-toggle right';
            rightToggle.innerHTML = '<i class="fas fa-folder"></i>';
            rightToggle.addEventListener('click', () => {
                toggleSidebar('.right-sidebar');
            });
            document.body.appendChild(rightToggle);
        }
    } else {
        // Remove mobile toggles if they exist
        const leftToggle = document.getElementById('leftSidebarToggle');
        const rightToggle = document.getElementById('rightSidebarToggle');
        
        if (leftToggle) leftToggle.remove();
        if (rightToggle) rightToggle.remove();
    }
}

// WebSocket handling
class ChatWebSocket {
    constructor(clientId) {
        this.clientId = clientId;
        this.socket = null;
        this.callbacks = {
            message: [],
            typing: [],
            done: [],
            conversations: [],
            history: [],
            files: [],
            progress: [],
            error: [],
            reconnect: []
        };
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectInterval = 3000;
    }
    
    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/${this.clientId}`;
        
        this.socket = new WebSocket(wsUrl);
        
        this.socket.onopen = () => {
            console.log('WebSocket connection established');
            this.reconnectAttempts = 0;
            this.trigger('reconnect');
        };
        
        this.socket.onmessage = (event) => {
            this.handleMessage(event.data);
        };
        
        this.socket.onclose = () => {
            console.log('WebSocket connection closed');
            this.attemptReconnect();
        };
        
        this.socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.attemptReconnect();
        };
    }
    
    handleMessage(data) {
        // Special message types
        if (data === "[DONE]") {
            this.trigger('done');
            return;
        }
        
        if (data === "[TYPING]") {
            this.trigger('typing');
            return;
        }
        
        // Handle conversation list
        if (data.startsWith("[CONVERSATIONS]")) {
            const conversationsJson = data.substring("[CONVERSATIONS]".length);
            try {
                const conversations = JSON.parse(conversationsJson);
                this.trigger('conversations', conversations);
            } catch (e) {
                console.error("Error parsing conversations:", e);
            }
            return;
        }
        
        // Handle conversation history
        if (data.startsWith("[HISTORY]")) {
            const historyJson = data.substring("[HISTORY]".length);
            try {
                const history = JSON.parse(historyJson);
                this.trigger('history', history);
            } catch (e) {
                console.error("Error parsing conversation history:", e);
            }
            return;
        }
        
        // Handle files list update
        if (data.startsWith("[FILES]")) {
            const filesJson = data.substring("[FILES]".length);
            try {
                const filesData = JSON.parse(filesJson);
                this.trigger('files', filesData.files);
            } catch (e) {
                console.error("Error parsing files data:", e);
            }
            return;
        }
        
        // Handle progress updates
        if (data.startsWith("[PROGRESS]")) {
            const progressJson = data.substring("[PROGRESS]".length);
            try {
                const progressData = JSON.parse(progressJson);
                this.trigger('progress', progressData);
            } catch (e) {
                console.error("Error parsing progress data:", e);
            }
            return;
        }
        
        // Handle errors
        if (data.startsWith("[ERROR]")) {
            const errorMessage = data.substring("[ERROR]".length);
            this.trigger('error', errorMessage);
            return;
        }
        
        // Regular messages
        this.trigger('message', data);
    }
    
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
            setTimeout(() => {
                this.connect();
            }, this.reconnectInterval);
        } else {
            console.error('Maximum reconnection attempts reached.');
        }
    }
    
    send(message) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(message);
            return true;
        } else {
            console.error('WebSocket is not connected.');
            showToast('Connection lost. Trying to reconnect...', 'error');
            this.connect();
            return false;
        }
    }
    
    on(event, callback) {
        if (this.callbacks[event]) {
            this.callbacks[event].push(callback);
        }
    }
    
    off(event, callback) {
        if (this.callbacks[event]) {
            this.callbacks[event] = this.callbacks[event].filter(cb => cb !== callback);
        }
    }
    
    trigger(event, data) {
        if (this.callbacks[event]) {
            this.callbacks[event].forEach(callback => callback(data));
        }
    }
}

// Chat UI Controller
class ChatUIController {
    constructor() {
        this.messageInput = document.getElementById('messageInput');
        this.chatForm = document.getElementById('chatForm');
        this.chatMessages = document.getElementById('chatMessages');
        this.filePreview = document.getElementById('filePreview');
        this.conversationTitle = document.getElementById('conversationTitle');
        this.saveTitleBtn = document.getElementById('saveTitleBtn');
        this.newConversationBtn = document.getElementById('newConversationBtn');
        this.conversationsList = document.getElementById('conversationsList');
        this.searchInput = document.getElementById('searchConversations');
        this.progressItems = document.getElementById('progressItems');
        this.fileList = document.getElementById('fileList');
        this.toastContainer = document.getElementById('toastContainer');
        
        this.currentConversationId = null;
        this.socket = null;
        this.files = new Map();
        
        this.setupEventListeners();
        this.initializeWebSocket();
        this.loadConversations();
    }

    setupEventListeners() {
        // Message input handling
        this.messageInput.addEventListener('keydown', (e) => {
            // Submit on Command+Enter (Mac) or Control+Enter (Windows/Linux)
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                e.preventDefault();
                this.chatForm.dispatchEvent(new Event('submit'));
                return;
            }
            
            // Add new line on Enter
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                const start = this.messageInput.selectionStart;
                const end = this.messageInput.selectionEnd;
                const value = this.messageInput.value;
                
                this.messageInput.value = value.substring(0, start) + '\n' + value.substring(end);
                this.messageInput.selectionStart = this.messageInput.selectionEnd = start + 1;
                
                // Trigger auto-resize
                this.messageInput.style.height = 'auto';
                this.messageInput.style.height = this.messageInput.scrollHeight + 'px';
            }
        });

        // Auto-resize textarea
        this.messageInput.addEventListener('input', () => {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = this.messageInput.scrollHeight + 'px';
        });

        // Form submission
        this.chatForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const message = this.messageInput.value.trim();
            if (!message && this.files.size === 0) return;

            await this.sendMessage(message);
            this.messageInput.value = '';
            this.messageInput.style.height = 'auto';
        });

        // New conversation
        this.newConversationBtn.addEventListener('click', () => {
            this.createNewConversation();
        });

        // Save conversation title
        this.saveTitleBtn.addEventListener('click', () => {
            this.saveConversationTitle();
        });

        // Search conversations
        this.searchInput.addEventListener('input', (e) => {
            this.searchConversations(e.target.value);
        });
    }

    initializeWebSocket() {
        this.socket = new WebSocket(`ws://${window.location.host}/ws`);
        
        this.socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };

        this.socket.onclose = () => {
            console.log('WebSocket connection closed');
            // Attempt to reconnect after a delay
            setTimeout(() => this.initializeWebSocket(), 3000);
        };
    }

    async sendMessage(message) {
        if (!this.currentConversationId) {
            await this.createNewConversation();
        }

        // Add user message to UI
        this.addMessage('user', message);

        // Prepare files if any
        const fileData = [];
        for (const [filename, file] of this.files.entries()) {
            const base64 = await this.fileToBase64(file);
            fileData.push({
                filename,
                content: base64,
                type: file.type
            });
        }

        // Clear file preview
        this.files.clear();
        this.filePreview.innerHTML = '';

        // Send message through WebSocket
        this.socket.send(JSON.stringify({
            type: 'message',
            conversation_id: this.currentConversationId,
            content: message,
            files: fileData
        }));

        // Show thinking indicator
        this.showThinkingIndicator();
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'bot_message':
                this.hideThinkingIndicator();
                this.addMessage('bot', data.content);
                break;
            case 'thinking_steps':
                this.updateThinkingSteps(data.content);
                break;
            case 'progress_update':
                this.updateProgress(data.content);
                break;
            case 'error':
                this.showError(data.content);
                break;
        }
    }

    addMessage(type, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `${type}-message`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = this.formatMessage(content);
        
        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = new Date().toLocaleTimeString();
        
        messageDiv.appendChild(contentDiv);
        messageDiv.appendChild(timeDiv);
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }

    formatMessage(content) {
        // Convert markdown to HTML
        return marked.parse(content);
    }

    showThinkingIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'thinking-steps';
        indicator.id = 'thinkingSteps';
        indicator.textContent = 'Thinking...';
        this.chatMessages.appendChild(indicator);
        this.scrollToBottom();
    }

    hideThinkingIndicator() {
        const indicator = document.getElementById('thinkingSteps');
        if (indicator) {
            indicator.remove();
        }
    }

    updateThinkingSteps(steps) {
        const indicator = document.getElementById('thinkingSteps');
        if (indicator) {
            indicator.textContent = steps;
            this.scrollToBottom();
        }
    }

    updateProgress(progress) {
        this.progressItems.innerHTML = '';
        
        progress.forEach(item => {
            const progressItem = document.createElement('div');
            progressItem.className = 'progress-item';
            
            const icon = document.createElement('i');
            icon.className = item.completed ? 'fas fa-check' : 'fas fa-circle';
            
            const text = document.createElement('span');
            text.textContent = item.description;
            
            progressItem.appendChild(icon);
            progressItem.appendChild(text);
            this.progressItems.appendChild(progressItem);
        });
    }

    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        this.chatMessages.appendChild(errorDiv);
        this.scrollToBottom();
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    async createNewConversation() {
        const response = await fetch('/api/conversations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (response.ok) {
            const data = await response.json();
            this.currentConversationId = data.id;
            this.conversationTitle.value = data.title || 'Untitled Conversation';
            this.loadConversations();
            this.showToast('New conversation created', 'success');
        }
    }

    async loadConversations() {
        const response = await fetch('/api/conversations');
        if (response.ok) {
            const conversations = await response.json();
            this.renderConversations(conversations);
        }
    }

    renderConversations(conversations) {
        this.conversationsList.innerHTML = '';
        
        conversations.forEach(conv => {
            const item = document.createElement('div');
            item.className = 'conversation-item';
            if (conv.id === this.currentConversationId) {
                item.classList.add('active');
            }
            
            const title = document.createElement('div');
            title.className = 'conversation-title';
            title.textContent = conv.title || 'Untitled Conversation';
            
            const date = document.createElement('div');
            date.className = 'conversation-date';
            date.textContent = new Date(conv.created_at).toLocaleDateString();
            
            item.appendChild(title);
            item.appendChild(date);
            
            item.addEventListener('click', () => this.loadConversation(conv.id));
            
            this.conversationsList.appendChild(item);
        });
    }

    async loadConversation(id) {
        const response = await fetch(`/api/conversations/${id}`);
        if (response.ok) {
            const conversation = await response.json();
            this.currentConversationId = id;
            this.conversationTitle.value = conversation.title;
            this.chatMessages.innerHTML = '';
            conversation.messages.forEach(msg => {
                this.addMessage(msg.role, msg.content);
            });
            this.loadConversations();
        }
    }

    async saveConversationTitle() {
        if (!this.currentConversationId) return;

        const response = await fetch(`/api/conversations/${this.currentConversationId}`, {
            method: 'PATCH',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                title: this.conversationTitle.value
            })
        });

        if (response.ok) {
            this.loadConversations();
            this.showToast('Conversation title saved', 'success');
        }
    }

    searchConversations(query) {
        const items = this.conversationsList.getElementsByClassName('conversation-item');
        for (const item of items) {
            const title = item.getElementsByClassName('conversation-title')[0].textContent;
            if (title.toLowerCase().includes(query.toLowerCase())) {
                item.style.display = 'flex';
            } else {
                item.style.display = 'none';
            }
        }
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        
        const content = document.createElement('div');
        content.className = 'toast-content';
        
        const icon = document.createElement('i');
        icon.className = `fas fa-${type === 'success' ? 'check-circle' : 'info-circle'}`;
        
        content.appendChild(icon);
        content.appendChild(document.createTextNode(message));
        toast.appendChild(content);
        
        this.toastContainer.appendChild(toast);
        
        // Trigger reflow
        toast.offsetHeight;
        
        toast.classList.add('show');
        
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    async fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result.split(',')[1]);
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }
}

// Initialize the chat UI when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.chatUI = new ChatUIController();
});

// Add CSS for toasts
(() => {
    const style = document.createElement('style');
    style.textContent = `
        .toast-container {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 9999;
        }
        
        .toast {
            background-color: white;
            border-radius: 8px;
            padding: 10px 15px;
            margin-top: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            transform: translateY(100px);
            opacity: 0;
            transition: all 0.3s ease;
        }
        
        .toast.show {
            transform: translateY(0);
            opacity: 1;
        }
        
        .toast-content {
            display: flex;
            align-items: center;
        }
        
        .toast-content i {
            margin-right: 10px;
        }
        
        .toast-success i {
            color: var(--success-color);
        }
        
        .toast-error i {
            color: var(--danger-color);
        }
        
        .toast-info i {
            color: var(--info-color);
        }
        
        .mobile-sidebar-toggle {
            position: fixed;
            bottom: 20px;
            z-index: 1001;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background-color: var(--primary-color);
            color: white;
            border: none;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
        }
        
        .mobile-sidebar-toggle.left {
            left: 20px;
        }
        
        .mobile-sidebar-toggle.right {
            right: 20px;
        }
    `;
    document.head.appendChild(style);
})(); 