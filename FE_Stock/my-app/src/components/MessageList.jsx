// src/components/MessageList.jsx
const MessageList = ({ messages }) => {
  return (
    <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
      {messages.map((msg, index) => (
        <div key={index} style={{ margin: '8px 0' }}>
          <strong>{msg.sender === 'admin' ? 'Admin' : 'User'}:</strong> {msg.content}
        </div>
      ))}
    </div>
  );
};

export default MessageList;
