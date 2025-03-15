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
    constructor(clientId) {
        this.clientId = clientId;
        this.ws = new ChatWebSocket(clientId);
        this.conversations = [];
        this.currentConversationId = null;
        this.uploadedFiles = [];
        this.modals = {};
        
        // Initialize
        this.initializeUI();
    }
    
    initializeUI() {
        // Set up marked.js for markdown rendering
        marked.setOptions({
            breaks: true,
            highlight: function (code, lang) {
                if (lang && hljs.getLanguage(lang)) {
                    try {
                        return hljs.highlight(code, { language: lang }).value;
                    } catch (e) {
                        console.error("Highlight error:", e);
                        return code;
                    }
                }
                return hljs.highlightAuto(code).value;
            },
            gfm: true
        });
        
        // Initialize modals
        this.modals.newConversation = new bootstrap.Modal(document.getElementById('newConversationModal'));
        
        // Connect WebSocket
        this.ws.connect();
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Set up WebSocket handlers
        this.setupWebSocketHandlers();
        
        // Update UI for mobile if needed
        setupMobileControls();
        window.addEventListener('resize', setupMobileControls);
    }
    
    setupEventListeners() {
        // Chat form submission
        document.getElementById('chatForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.sendMessage();
        });
        
        // New conversation button
        document.getElementById('newConversationBtn').addEventListener('click', () => {
            this.modals.newConversation.show();
        });
        
        // Start conversation button (empty state)
        document.getElementById('startConversationBtn').addEventListener('click', () => {
            this.modals.newConversation.show();
        });
        
        // Create conversation button in modal
        document.getElementById('createConversationBtn').addEventListener('click', () => {
            const title = document.getElementById('newConversationTitle').value || "Financial Planning Session";
            this.ws.send(`/new ${title}`);
            this.modals.newConversation.hide();
        });
        
        // Save conversation title button
        document.getElementById('saveConversationTitle').addEventListener('click', () => {
            if (this.currentConversationId) {
                const title = document.getElementById('conversationTitle').value;
                if (title) {
                    this.ws.send(`/rename ${title}`);
                }
            }
        });
        
        // File upload
        document.getElementById('fileInput').addEventListener('change', this.handleFileUpload.bind(this));
        
        // Auto-resize message input
        const messageInput = document.getElementById('messageInput');
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
        
        // Conversation search
        document.getElementById('conversationSearch').addEventListener('input', (e) => {
            this.filterConversations(e.target.value);
        });
    }
    
    setupWebSocketHandlers() {
        // Regular message
        this.ws.on('message', (data) => {
            // Handle thinking steps differently
            if (data.startsWith("🤔 Thinking Process:")) {
                this.addThinkingSteps(data);
            } else {
                this.addBotMessage(data);
            }
        });
        
        // Typing indicator
        this.ws.on('typing', () => {
            document.getElementById('typingIndicator').style.display = 'block';
            this.scrollToBottom();
        });
        
        // Done typing
        this.ws.on('done', () => {
            document.getElementById('typingIndicator').style.display = 'none';
            this.scrollToBottom();
        });
        
        // Conversations list
        this.ws.on('conversations', (conversations) => {
            this.conversations = conversations;
            this.updateConversationsList();
        });
        
        // Conversation history
        this.ws.on('history', (history) => {
            this.loadConversationHistory(history);
        });
        
        // Files list
        this.ws.on('files', (files) => {
            this.updateFilesList(files);
        });
        
        // Progress updates
        this.ws.on('progress', (progressData) => {
            this.updateProgressTracker(progressData);
        });
        
        // Error messages
        this.ws.on('error', (message) => {
            this.showError(message);
            showToast(message, 'error');
        });
        
        // Reconnect
        this.ws.on('reconnect', () => {
            // Request available conversations
            this.ws.send("/list");
        });
    }
    
    sendMessage() {
        const messageInput = document.getElementById('messageInput');
        const message = messageInput.value.trim();
        
        if (!message) return;
        
        // Clear input
        messageInput.value = '';
        messageInput.style.height = 'auto';
        
        // Add message to chat
        this.addUserMessage(message);
        
        // Send via WebSocket
        if (this.uploadedFiles.length > 0) {
            // Send message with file references
            const fileData = {
                message: message,
                paths: this.uploadedFiles.map(f => f.path)
            };
            this.ws.send(`FILE:${JSON.stringify(fileData)}`);
            
            // Clear uploaded files
            this.uploadedFiles = [];
            document.getElementById('filePreview').innerHTML = '';
        } else {
            // Send regular message
            this.ws.send(message);
        }
    }
    
    addUserMessage(message) {
        const messagesContainer = document.getElementById('chatMessages');
        const messageElement = document.createElement('div');
        messageElement.className = 'user-message';
        
        const time = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        
        messageElement.innerHTML = `
            <div class="message-content">${message}</div>
            <div class="message-time">${time}</div>
        `;
        
        messagesContainer.appendChild(messageElement);
        this.scrollToBottom();
    }
    
    addBotMessage(message) {
        const messagesContainer = document.getElementById('chatMessages');
        const messageElement = document.createElement('div');
        messageElement.className = 'bot-message';
        
        const time = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        
        // Convert markdown to HTML
        const htmlContent = marked.parse(message);
        
        messageElement.innerHTML = `
            <div class="message-content">${htmlContent}</div>
            <div class="message-time">${time}</div>
        `;
        
        // Apply syntax highlighting to code blocks
        messageElement.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });
        
        messagesContainer.appendChild(messageElement);
        this.scrollToBottom();
    }
    
    addThinkingSteps(steps) {
        const messagesContainer = document.getElementById('chatMessages');
        const stepsElement = document.createElement('div');
        stepsElement.className = 'thinking-steps';
        
        stepsElement.textContent = steps;
        
        messagesContainer.appendChild(stepsElement);
        this.scrollToBottom();
    }
    
    showError(message) {
        const messagesContainer = document.getElementById('chatMessages');
        const errorElement = document.createElement('div');
        errorElement.className = 'error-message';
        
        errorElement.textContent = message;
        
        messagesContainer.appendChild(errorElement);
        this.scrollToBottom();
    }
    
    scrollToBottom() {
        const messagesContainer = document.getElementById('chatMessages');
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    updateConversationsList() {
        const list = document.getElementById('conversationsList');
        
        // Show/hide empty state
        const emptyState = list.querySelector('.conversation-empty-state');
        if (emptyState) {
            emptyState.style.display = this.conversations.length === 0 ? 'block' : 'none';
        }
        
        // Clear existing conversations (except empty state)
        const existingItems = list.querySelectorAll('.conversation-item');
        existingItems.forEach(item => item.remove());
        
        // Add conversations
        this.conversations.forEach(conversation => {
            const item = document.createElement('div');
            item.className = 'conversation-item';
            if (conversation.id === this.currentConversationId) {
                item.classList.add('active');
            }
            
            // Format date
            const formattedDate = formatDate(conversation.last_updated);
            
            item.innerHTML = `
                <div class="conversation-info">
                    <div class="conversation-title">${conversation.title}</div>
                    <div class="conversation-date">${formattedDate}</div>
                </div>
            `;
            
            // Add click handler to load conversation
            item.addEventListener('click', () => {
                this.ws.send(`/load ${conversation.id}`);
                this.currentConversationId = conversation.id;
                document.getElementById('conversationTitle').value = conversation.title;
                this.updateConversationsList(); // Update active state
            });
            
            list.appendChild(item);
        });
        
        // Update conversation title input
        if (this.currentConversationId) {
            const currentConv = this.conversations.find(c => c.id === this.currentConversationId);
            if (currentConv) {
                document.getElementById('conversationTitle').value = currentConv.title;
            }
        }
    }
    
    filterConversations(searchText) {
        if (!searchText) {
            // Show all conversations
            const items = document.querySelectorAll('.conversation-item');
            items.forEach(item => item.style.display = 'flex');
            return;
        }
        
        // Filter by title (case insensitive)
        const search = searchText.toLowerCase();
        const items = document.querySelectorAll('.conversation-item');
        
        items.forEach(item => {
            const title = item.querySelector('.conversation-title').textContent.toLowerCase();
            item.style.display = title.includes(search) ? 'flex' : 'none';
        });
    }
    
    loadConversationHistory(history) {
        // Clear chat messages
        document.getElementById('chatMessages').innerHTML = '';
        
        // Add history messages
        history.forEach(message => {
            if (message.role === 'user') {
                this.addUserMessage(message.content);
            } else if (message.role === 'assistant') {
                this.addBotMessage(message.content);
            } else if (message.role === 'system' && message.content.startsWith('🤔 Thinking Process:')) {
                this.addThinkingSteps(message.content);
            }
        });
        
        // Add typing indicator (hidden by default)
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'typing-indicator';
        typingIndicator.id = 'typingIndicator';
        typingIndicator.innerHTML = '<span></span><span></span><span></span>';
        typingIndicator.style.display = 'none';
        document.getElementById('chatMessages').appendChild(typingIndicator);
    }
    
    handleFileUpload(event) {
        const files = event.target.files;
        if (!files || files.length === 0) return;
        
        const filePreview = document.getElementById('filePreview');
        const uploadProgress = document.getElementById('uploadProgress');
        
        // Show progress bar
        uploadProgress.style.display = 'block';
        
        // Create FormData
        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append('files', files[i]);
        }
        
        // Upload files
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/upload/');
        
        xhr.upload.onprogress = function(e) {
            if (e.lengthComputable) {
                const percentComplete = (e.loaded / e.total) * 100;
                uploadProgress.querySelector('.progress-bar').style.width = percentComplete + '%';
            }
        };
        
        xhr.onload = function() {
            if (xhr.status === 200) {
                const response = JSON.parse(xhr.responseText);
                
                // Add files to preview
                response.file_paths.forEach((path, index) => {
                    const file = files[index];
                    const fileItem = document.createElement('div');
                    fileItem.className = 'file-item';
                    
                    fileItem.innerHTML = `
                        ${file.name}
                        <button type="button" class="remove-file" data-path="${path}">
                            <i class="fas fa-times"></i>
                        </button>
                    `;
                    
                    const removeBtn = fileItem.querySelector('.remove-file');
                    removeBtn.addEventListener('click', function() {
                        this.uploadedFiles = this.uploadedFiles.filter(f => f.path !== path);
                        fileItem.remove();
                    }.bind(this));
                    
                    filePreview.appendChild(fileItem);
                    
                    // Add to uploaded files
                    this.uploadedFiles.push({
                        name: file.name,
                        path: path
                    });
                }, this);
                
                // Hide progress bar
                setTimeout(() => {
                    uploadProgress.style.display = 'none';
                    uploadProgress.querySelector('.progress-bar').style.width = '0%';
                }, 500);
            } else {
                console.error('Upload failed');
                uploadProgress.style.display = 'none';
                showToast('File upload failed', 'error');
            }
        }.bind(this);
        
        xhr.send(formData);
    },
    
    updateFilesList(files) {
        const fileList = document.getElementById('fileList');
        const emptyMessage = document.getElementById('emptyFilesMessage');
        
        // Show/hide empty message
        emptyMessage.style.display = files.length === 0 ? 'block' : 'none';
        
        // Clear existing files
        fileList.innerHTML = '';
        
        // Add files
        files.forEach(file => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            
            // Choose icon based on file type
            let icon = 'fa-file';
            if (file.type === 'markdown') icon = 'fa-file-alt';
            else if (file.type === 'website') icon = 'fa-globe';
            else if (file.type === 'report') icon = 'fa-chart-bar';
            else if (file.type === 'document') icon = 'fa-file-pdf';
            else if (file.type === 'text') icon = 'fa-file-alt';
            
            fileItem.innerHTML = `
                <div class="file-icon">
                    <i class="fas ${icon}"></i>
                </div>
                <div class="file-info">
                    <div class="file-name">${file.name}</div>
                    <div class="file-meta">${file.type}</div>
                </div>
                <div class="file-actions">
                    <button class="file-action-btn download-btn" data-path="${file.path}" title="Download">
                        <i class="fas fa-download"></i>
                    </button>
                    <button class="file-action-btn view-btn" data-path="${file.path}" title="View">
                        <i class="fas fa-eye"></i>
                    </button>
                </div>
            `;
            
            // Add event listeners for buttons
            fileItem.querySelector('.download-btn').addEventListener('click', function() {
                window.open(`/download/${encodeURIComponent(file.name)}`, '_blank');
            });
            
            fileItem.querySelector('.view-btn').addEventListener('click', function() {
                if (file.type === 'website') {
                    window.open(`/view-website/${encodeURIComponent(file.path)}`, '_blank');
                } else {
                    window.open(`/download/${encodeURIComponent(file.name)}`, '_blank');
                }
            });
            
            fileList.appendChild(fileItem);
        });
    }
    
    updateProgressTracker(progressData) {
        const progressTracker = document.getElementById('progressTracker');
        const sections = progressData.sections || [];
        const currentSection = progressData.current_section;
        const completedTasks = progressData.completed_tasks || [];
        
        // Clear progress tracker
        progressTracker.innerHTML = '';
        
        // Show empty state if no sections
        if (sections.length === 0) {
            progressTracker.innerHTML = `
                <div class="empty-progress-state">
                    <p>No active project</p>
                </div>
            `;
            return;
        }
        
        // Add sections
        sections.forEach(section => {
            const sectionElement = document.createElement('div');
            sectionElement.className = 'progress-section';
            
            // Determine section status
            let status = 'incomplete';
            if (section === currentSection) status = 'in-progress';
            if (completedTasks.includes(section)) status = 'complete';
            
            // Choose icon based on section
            let icon = 'fa-list-check';
            if (section.toLowerCase().includes('research')) icon = 'fa-search';
            else if (section.toLowerCase().includes('analysis')) icon = 'fa-chart-line';
            else if (section.toLowerCase().includes('document')) icon = 'fa-file-alt';
            else if (section.toLowerCase().includes('report')) icon = 'fa-file-pdf';
            
            sectionElement.innerHTML = `
                <div class="progress-section-header">
                    <div class="progress-section-title">
                        <i class="fas ${icon}"></i>
                        ${section}
                    </div>
                    <div class="progress-section-indicator ${status}">
                        <i class="fas ${status === 'complete' ? 'fa-check' : status === 'in-progress' ? 'fa-spinner fa-spin' : 'fa-circle'}"></i>
                    </div>
                </div>
            `;
            
            progressTracker.appendChild(sectionElement);
        });
        
        // Add completed tasks
        if (completedTasks.length > 0) {
            const tasksSection = document.createElement('div');
            tasksSection.className = 'progress-section-tasks';
            
            completedTasks.forEach(task => {
                const taskElement = document.createElement('div');
                taskElement.className = 'progress-task task-complete';
                taskElement.innerHTML = `
                    <i class="fas fa-check-circle"></i>
                    ${task}
                `;
                tasksSection.appendChild(taskElement);
            });
            
            progressTracker.appendChild(tasksSection);
        }
    }
}

// Initialize everything when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Generate a unique client ID for this session
    const clientId = 'client_' + Math.random().toString(36).substring(2, 10);
    
    // Initialize chat controller
    const chatController = new ChatUIController(clientId);
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