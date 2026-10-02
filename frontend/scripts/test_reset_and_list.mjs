const res = await fetch('http://127.0.0.1:9222/json');
const tabs = await res.json();
const tab = tabs.find(t => t.url && t.url.includes('localhost:3000')) || tabs[0];
const ws = new WebSocket(tab.webSocketDebuggerUrl);

ws.onopen = () => {
  ws.send(JSON.stringify({ id: 1, method: 'Runtime.enable' }));
  ws.send(JSON.stringify({ id: 2, method: 'Console.enable' }));

  setTimeout(() => {
    // Click "Analyze Another Recording"
    ws.send(JSON.stringify({
      id: 10,
      method: 'Runtime.evaluate',
      params: {
        expression: `
          (() => {
            const btns = Array.from(document.querySelectorAll('button'));
            const resetBtn = btns.find(b => b.innerText.includes('Analyze Another Recording') || b.innerText.includes('New Analysis'));
            if (resetBtn) {
              resetBtn.click();
              return 'CLICKED_RESET';
            }
            return 'RESET_NOT_FOUND';
          })()
        `,
        returnByValue: true
      }
    }));
  }, 500);

  setTimeout(() => {
    // Check buttons after reset
    ws.send(JSON.stringify({
      id: 20,
      method: 'Runtime.evaluate',
      params: {
        expression: `
          (() => {
            return Array.from(document.querySelectorAll('button')).map((b, i) => ({
              index: i,
              text: b.innerText.trim().replace(/\\n/g, ' '),
              disabled: b.disabled,
              visible: b.offsetWidth > 0 && b.offsetHeight > 0
            }));
          })()
        `,
        returnByValue: true
      }
    }));
  }, 1200);

  setTimeout(() => {
    ws.close();
    process.exit(0);
  }, 2500);
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.id === 10) {
    console.log('CLICK RESULT:', msg.result?.result?.value);
  } else if (msg.id === 20) {
    console.log('RESET PAGE BUTTONS:');
    console.log(JSON.stringify(msg.result?.result?.value, null, 2));
  }
};
