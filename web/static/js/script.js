function setTask(taskName) {
    // 1. Update UI immediately for responsiveness
    document.querySelectorAll('.task-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`btn-${taskName}`).classList.add('active');

    // 2. Prepare payload
    const payload = {
        task: taskName
    };

    // 3. Send to Backend
    sendUpdate(payload);
}

function updateSettings() {
    const confValue = document.getElementById('conf-slider').value;
    const modelValue = document.getElementById('model-select').value;

    // Update Label
    document.getElementById('conf-value').innerText = parseFloat(confValue).toFixed(2);

    const payload = {
        conf: confValue,
        model: modelValue
    };

    sendUpdate(payload);
}

function sendUpdate(payload) {
    fetch('/update_settings', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
    })
    .then(response => response.json())
    .then(data => {
        // Update Status Info
        document.getElementById('current-task').innerText = data.task.charAt(0).toUpperCase() + data.task.slice(1);
        document.getElementById('current-model').innerText = data.model;
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

function shutdownApp() {
    if(confirm("¿Estás seguro de que quieres cerrar la aplicación?")) {
        fetch('/shutdown', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            alert("Servidor detenido. Puedes cerrar esta pestaña.");
            document.body.innerHTML = "<h1 style='color:white; text-align:center; margin-top:20%'>Servidor Detenido</h1>";
        });
    }
}

// Initial update on load to sync labels
document.addEventListener('DOMContentLoaded', () => {
   // Optional: Fetch initial state from server if needed
});
