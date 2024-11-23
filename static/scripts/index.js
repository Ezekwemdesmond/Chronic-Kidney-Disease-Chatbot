document.getElementById('send').addEventListener('click', function () {
    let input = document.getElementById('user-input');
    let message = input.value.trim();

    if (!message) return;

    // Get current time
    const timestamp = new Date().toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit'
    });

    // Append user message
    let chatbox = document.getElementById('messages');
    let userDiv = document.createElement('div');
    userDiv.className = 'message user-message';
    userDiv.innerHTML = `
        <div class="message-label">You:</div>
        <div class="message-content">${message}</div>
        <div class="message-timestamp">${timestamp}</div>
    `;
    chatbox.appendChild(userDiv);

    // Add "typing..." indicator
    let typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message typing-indicator';
    typingDiv.innerHTML = `
        <div class="message-label">
            <img src="/static/images/bot-icon.png" alt="Bot Icon" class="bot-icon">
            KidneyCareAI
        </div>
        <div class="message-content">Typing...</div>
    `;
    chatbox.appendChild(typingDiv);

    // Scroll to bottom
    chatbox.scrollTop = chatbox.scrollHeight;

    // Make the fetch request
    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message }),
    })
        .then(response => response.json())
        .then(data => {
            // Remove "typing..." indicator
            chatbox.removeChild(typingDiv);

            // Append bot response with timestamp
            let botDiv = document.createElement('div');
            botDiv.className = 'message bot-message';
            botDiv.innerHTML = `
                <div class="message-label">
                    <img src="/static/images/bot-icon.png" alt="Bot Icon" class="bot-icon">
                    KidneyCareAI
                </div>
                <div class="message-content">${data.response}</div>
                <div class="message-timestamp">${timestamp}</div>
            `;
            chatbox.appendChild(botDiv);

            // Scroll to bottom
            chatbox.scrollTop = chatbox.scrollHeight;
        })
        .catch(() => {
            // Remove "typing..." indicator
            chatbox.removeChild(typingDiv);

            // Append error message
            let botDiv = document.createElement('div');
            botDiv.className = 'message bot-message';
            botDiv.innerHTML = `
                <div class="message-label">
                    <img src="/static/images/bot-icon.png" alt="Bot Icon" class="bot-icon">
                    KidneyCareAI
                </div>
                <div class="message-content">Sorry, something went wrong. Please try again later.</div>
            `;
            chatbox.appendChild(botDiv);

            // Scroll to bottom
            chatbox.scrollTop = chatbox.scrollHeight;
        });

    // Clear input field
    input.value = '';
});

// Allow sending message with Enter key
document.getElementById('user-input').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        document.getElementById('send').click();
    }
});
