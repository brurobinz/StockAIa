// src/components/MessageInput.jsx
import React, { useState } from 'react';

const MessageInput = ({ onSend }) => {
  const [text, setText] = useState('');

  const handleSend = () => {
    if (text.trim()) {
      onSend(text);
      setText('');
    }
  };

  return (
    <div style={{ marginTop: '10px' }}>
      <input
        type="text"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter your message..."
        style={{ width: '70%' }}
      />
      <button onClick={handleSend} style={{ marginLeft: '10px' }}>Send</button>
    </div>
  );
};

export default MessageInput;
