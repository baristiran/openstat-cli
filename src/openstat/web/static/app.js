// OpenStat Web Client
(function() {
    let ws = null;
    let sessionId = null;
    const output = document.getElementById('output');
    const input = document.getElementById('command-input');
    const status = document.getElementById('status');
    const shapeInfo = document.getElementById('shape-info');
    const plotContainer = document.getElementById('plot-container');
    const plotImg = document.getElementById('plot-img');
    const fileUpload = document.getElementById('file-upload');
    const history = [];
    let historyIdx = -1;

    async function init() {
        const resp = await fetch('/api/session', { method: 'POST' });
        const data = await resp.json();
        sessionId = data.session_id;
        connectWS();
    }

    function connectWS() {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${proto}//${location.host}/ws/${sessionId}`);

        ws.onopen = () => {
            status.textContent = 'Connected';
            status.style.color = '#a6e3a1';
            appendOutput('OpenStat v0.3.0 — Web Interface\nType help for commands.\n', 'result');
        };

        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.content) {
                appendOutput(msg.content + '\n', 'result');
            }
            if (msg.shape) {
                shapeInfo.textContent = msg.shape;
            }
            if (msg.plot) {
                plotImg.src = 'data:image/png;base64,' + msg.plot;
                plotContainer.style.display = 'block';
            }
            if (msg.type === 'quit') {
                status.textContent = 'Disconnected';
                status.style.color = '#f38ba8';
            }
        };

        ws.onclose = () => {
            status.textContent = 'Disconnected';
            status.style.color = '#f38ba8';
        };
    }

    function appendOutput(text, cls) {
        const span = document.createElement('span');
        span.className = cls;
        span.textContent = text;
        output.appendChild(span);
        output.scrollTop = output.scrollHeight;
    }

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            const cmd = input.value.trim();
            if (!cmd) return;
            history.push(cmd);
            historyIdx = history.length;
            appendOutput('openstat> ' + cmd + '\n', 'cmd');
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(cmd);
            }
            input.value = '';
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            if (historyIdx > 0) {
                historyIdx--;
                input.value = history[historyIdx];
            }
        } else if (e.key === 'ArrowDown') {
            e.preventDefault();
            if (historyIdx < history.length - 1) {
                historyIdx++;
                input.value = history[historyIdx];
            } else {
                historyIdx = history.length;
                input.value = '';
            }
        }
    });

    fileUpload.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const formData = new FormData();
        formData.append('file', file);
        try {
            const resp = await fetch(`/api/upload/${sessionId}`, {
                method: 'POST',
                body: formData,
            });
            const data = await resp.json();
            if (data.result) {
                appendOutput(data.result + '\n', 'result');
            }
            if (data.shape) {
                shapeInfo.textContent = data.shape;
            }
        } catch (err) {
            appendOutput('Upload failed: ' + err + '\n', 'error');
        }
        fileUpload.value = '';
    });

    init();
})();
