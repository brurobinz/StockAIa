import React, { useEffect, useState } from 'react';
import axios from 'axios';
import '../pages/AdminHelp.css';

const AdminHelpPage = () => {
    const [messagesByUser, setMessagesByUser] = useState({});
    const [selectedUserId, setSelectedUserId] = useState(null);
    const [replyContent, setReplyContent] = useState('');

    const fetchAllMessages = async () => {
        try {
            const res = await axios.get('http://localhost:4000/api/message');
            const grouped = res.data.reduce((acc, msg) => {
                acc[msg.userId] = acc[msg.userId] || [];
                acc[msg.userId].push(msg);
                return acc;
            }, {});
            setMessagesByUser(grouped);
        } catch (err) {
            console.error(err);
        }
    };

    useEffect(() => {
        fetchAllMessages();
    }, []);

    const handleReply = async () => {
        if (!replyContent.trim()) return;
        try {
            await axios.post('http://localhost:4000/api/message', {
                sender: 'admin',
                userId: selectedUserId,
                content: replyContent
            });
            setReplyContent('');
            fetchAllMessages(); // cập nhật lại danh sách tin nhắn
        } catch (err) {
            console.error(err);
        }
    };

    return (
        <div className="admin-help-container">
            <div className="user-list">
                <h2>User List</h2>
                <ul>
                    {Object.keys(messagesByUser).map(userId => (
                        <li
                            key={userId}
                            onClick={() => setSelectedUserId(userId)}
                            className={selectedUserId === userId ? 'selected' : ''}
                        >
                            User ID: {userId}
                        </li>
                    ))}
                </ul>
            </div>

            <div className="message-viewer">
                <h2>Messages</h2>
                {selectedUserId && (
                    <>
                        <div className="messages">
                            {messagesByUser[selectedUserId].map((msg, index) => (
                                <div key={index} className={`message ${msg.sender}`}>
                                    <strong>{msg.sender}</strong>: {msg.content}
                                </div>
                            ))}
                        </div>
                        <div className="reply-box">
                            <textarea
                                value={replyContent}
                                onChange={(e) => setReplyContent(e.target.value)}
                                placeholder="Enter your reply..."
                            />
                            <button onClick={handleReply}>Send Reply</button>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
};

export default AdminHelpPage;
