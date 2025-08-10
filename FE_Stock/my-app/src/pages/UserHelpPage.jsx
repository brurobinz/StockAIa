import React, { useState, useEffect } from 'react';
import axios from 'axios';
import '../pages/UserHelpPage.css';
const UserHelpPage = () => {
    const [content, setContent] = useState('');
    const [messages, setMessages] = useState([]);
    const userId = localStorage.getItem('userId'); // hoặc lấy từ context, redux...

    const fetchMessages = async () => {
        try {
            const res = await axios.get(`http://localhost:4000/api/message/${userId}`);
            setMessages(res.data);
        } catch (err) {
            console.error('Error fetching messages:', err);
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!content.trim()) return;

        try {
            await axios.post('http://localhost:4000/api/message', {
                sender: 'user',
                userId,
                content
            });
            setContent('');
            fetchMessages(); // refresh messages
        } catch (err) {
            console.error('Error sending message:', err);
        }
    };

    useEffect(() => {
        fetchMessages();
    }, []);

    return (
        <div className="user-help-container">
    <h2>Support Team</h2>

    <form onSubmit={handleSubmit} className="user-help-form">
        <textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
            rows="4"
            placeholder="Enter content to admin ..."
        />
        <button type="submit">Gửi</button>
    </form>

    <div className="message-history">
        <h3>Response History:</h3>
        {messages.map((msg, index) => (
            <div
                key={index}
                className={`message-item ${msg.sender === 'admin' ? 'admin' : 'user'}`}
            >
                <span className="message-sender">
                    {msg.sender === 'admin' ? 'Admin' : 'You'}
                </span>
                {msg.content}
            </div>
        ))}
    </div>
</div>

    );
};

export default UserHelpPage;
