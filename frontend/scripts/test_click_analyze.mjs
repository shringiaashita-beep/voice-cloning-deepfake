const res = await fetch('http://127.0.0.1:9222/json');
const tabs = await res.json();
const tab = tabs.find(t => t.url && t.url.includes('localhost:3000')) || tabs[0];
const ws = new WebSocket(tab.webSocketDebuggerUrl);

ws.onopen = () => {
  ws.send(JSON.stringify({ id: 1, method: 'Runtime.enable' }));
  ws.send(JSON.stringify({ id: 2, method: 'Network.enable' }));
  ws.send(JSON.stringify({ id: 3, method: 'Console.enable' }));

  setTimeout(() => {
    console.log('Clicking Analyze Audio button...');
    ws.send(JSON.stringify({
      id: 10,
      method: 'Runtime.evaluate',
      params: {
        expression: `
          (() => {
            const btns = Array.from(document.querySelectorAll('button'));
            const analyzeBtn = btns.find(b => b.innerText.includes('Analyze Audio'));
            if (!analyzeBtn) return 'ANALYZE_BTN_NOT_FOUND';
            analyzeBtn.click();
            return 'CLICKED_ANALYZE_AUDIO';
          })()
        `,
        returnByValue: true
      }
    }));
  }, 500);

  setTimeout(() => {
    ws.send(JSON.stringify({
      id: 20,
      method: 'Runtime.evaluate',
      params: {
        expression: `
          (() => {
            return {
              bodyPreview: document.body.innerText.slice(0, 300),
              hasResults: document.body.innerText.includes('Deepfake Risk Meter') || document.body.innerText.includes('Forensic Signal Analysis')
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
  if (msg.method === 'Network.requestWillBeSent') {
    console.log('NETWORK REQUEST:', msg.params.request.method, msg.params.request.url);
  } else if (msg.method === 'Network.responseReceived') {
    console.log('NETWORK RESPONSE:', msg.params.response.status, msg.params.response.url);
  } else if (msg.method === 'Runtime.exceptionThrown') {
    console.error('EXCEPTION:', JSON.stringify(msg.params.exceptionDetails));
  } else if (msg.id === 10) {
    console.log('CLICK RESULT:', msg.result?.result?.value);
  } else if (msg.id === 20) {
    console.log('STATE AFTER CLICK:', JSON.stringify(msg.result?.result?.value, null, 2));
  }
};
