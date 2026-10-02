const res = await fetch('http://127.0.0.1:9222/json');
const tabs = await res.json();
const tab = tabs.find(t => t.url && t.url.includes('localhost:3000')) || tabs[0];
const ws = new WebSocket(tab.webSocketDebuggerUrl);

ws.onopen = () => {
  ws.send(JSON.stringify({ id: 1, method: 'Runtime.enable' }));
  ws.send(JSON.stringify({ id: 2, method: 'Network.enable' }));
  ws.send(JSON.stringify({ id: 3, method: 'Console.enable' }));

  setTimeout(() => {
    console.log('Simulating file selection on the file input...');
    ws.send(JSON.stringify({
      id: 10,
      method: 'Runtime.evaluate',
      params: {
        expression: `
          (async () => {
            const fileInput = document.querySelector('input[type="file"]');
            if (!fileInput) return 'INPUT_NOT_FOUND';
            
            // Create a realistic WAV file blob in browser
            const duration = 1.0;
            const sampleRate = 16000;
            const numFrames = sampleRate * duration;
            const buf = new ArrayBuffer(44 + numFrames * 2);
            const view = new DataView(buf);
            
            // RIFF header
            view.setUint32(0, 0x52494646, false); // "RIFF"
            view.setUint32(4, 36 + numFrames * 2, true);
            view.setUint32(8, 0x57415645, false); // "WAVE"
            view.setUint32(12, 0x666d7420, false); // "fmt "
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true); // PCM
            view.setUint16(22, 1, true); // Mono
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * 2, true);
            view.setUint16(32, 2, true);
            view.setUint16(34, 16, true);
            view.setUint32(36, 0x64617461, false); // "data"
            view.setUint32(40, numFrames * 2, true);
            
            for (let i = 0; i < numFrames; i++) {
              const val = Math.sin(2 * Math.PI * 440 * (i / sampleRate)) * 16000;
              view.setInt16(44 + i * 2, val, true);
            }
            
            const file = new File([buf], 'test_user_upload.wav', { type: 'audio/wav' });
            
            // Dispatch DataTransfer to file input
            const dt = new DataTransfer();
            dt.items.add(file);
            fileInput.files = dt.files;
            fileInput.dispatchEvent(new Event('change', { bubbles: true }));
            
            return {
              filesLength: fileInput.files.length,
              fileName: fileInput.files[0].name
            };
          })()
        `,
        awaitPromise: true,
        returnByValue: true
      }
    }));
  }, 1000);

  setTimeout(() => {
    // Check page state after file input change
    ws.send(JSON.stringify({
      id: 20,
      method: 'Runtime.evaluate',
      params: {
        expression: `
          (() => {
            const btns = Array.from(document.querySelectorAll('button')).map(b => b.innerText.trim().replace(/\\n/g, ' '));
            const hasAnalyzeBtn = btns.some(t => t.includes('Analyze Audio'));
            const hasLoading = !!document.querySelector('.loading') || document.body.innerText.includes('Uploading') || document.body.innerText.includes('Analyzing');
            const hasResults = document.body.innerText.includes('Deepfake Risk Meter') || document.body.innerText.includes('Court-Admissible Police Forensic Dossier');
            return {
              hasAnalyzeBtn,
              hasLoading,
              hasResults,
              buttons: btns
            };
          })()
        `,
        returnByValue: true
      }
    }));
  }, 2500);

  setTimeout(() => {
    ws.close();
    process.exit(0);
  }, 4000);
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.method === 'Console.messageAdded') {
    console.log('BROWSER CONSOLE:', msg.params.message.text);
  } else if (msg.method === 'Network.requestWillBeSent') {
    console.log('NETWORK REQUEST:', msg.params.request.method, msg.params.request.url);
  } else if (msg.id === 10) {
    console.log('FILE INPUT SET RESULT:', msg.result?.result?.value);
  } else if (msg.id === 20) {
    console.log('PAGE STATE AFTER FILE SELECTION:');
    console.log(JSON.stringify(msg.result?.result?.value, null, 2));
  }
};
