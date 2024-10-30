document.getElementById('send').addEventListener('click', function() {
    let input = document.getElementById('user-input');
    let message = input.value.trim();
    
    if (!message) return;

    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({message}),
    }).then(response => response.json()).then(data => {
        let chatbox = document.getElementById('messages');
        
        // Add user message
        let userDiv = document.createElement('div');
        userDiv.className = 'message user-message';
        userDiv.innerHTML = `
            <div class="message-label">You:</div>
            <div class="message-content">${message}</div>
        `;
        chatbox.appendChild(userDiv);
        
        // Add bot message
        let botDiv = document.createElement('div');
        botDiv.className = 'message bot-message';
        botDiv.innerHTML = `
            <div class="message-label">
                <i class="fas fa-user-md bot-icon"></i>
                Bot:
            </div>
            <div class="message-content">${data.response}</div>
        `;
        chatbox.appendChild(botDiv);
        
        // Clear input and scroll to bottom
        input.value = '';
        chatbox.scrollTop = chatbox.scrollHeight;
    });
});

// Allow sending message with Enter key
document.getElementById('user-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        document.getElementById('send').click();
    }
});