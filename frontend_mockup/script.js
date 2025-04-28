// script.js

// Recommendation logic
function recommendContent() {
    let text = document.getElementById('textInput').value.toLowerCase();
    if (text.includes('weak') || text.includes('low')) {
        document.getElementById('recommendationOutput').innerText = "Recommendation: Easier Content";
    } else if (text.includes('excellent') || text.includes('high')) {
        document.getElementById('recommendationOutput').innerText = "Recommendation: Advanced Content";
    } else {
        document.getElementById('recommendationOutput').innerText = "Recommendation: Keep Current Level";
    }
}

// Chatbot interaction
function sendMessage() {
    let userMessage = document.getElementById('chatInput').value;
    let chatHistory = document.getElementById('chatHistory');
    if (userMessage.trim() !== "") {
        chatHistory.innerHTML += `<p><strong>You:</strong> ${userMessage}</p>`;
        chatHistory.innerHTML += `<p><strong>Bot:</strong> Thanks for your message!</p>`;
        document.getElementById('chatInput').value = '';
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
}

// Progress Chart using Chart.js
const ctx = document.getElementById('progressChart').getContext('2d');
const progressChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
        datasets: [{
            label: 'Student Progress',
            data: [60, 65, 70, 80],
            borderColor: 'rgba(75, 192, 192, 1)',
            tension: 0.4
        }]
    },
    options: {
        responsive: true
    }
});
