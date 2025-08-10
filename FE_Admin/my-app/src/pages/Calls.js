// pages/Calls.js
import React, { useState } from 'react';
import './Calls.css';

const mockUsers = [
  { id: 1, name: 'Nguyễn Văn Cường', phone: '0901234567' },
  { id: 2, name: 'Phạm Hùng Cường', phone: '0912345678' },
  { id: 3, name: 'Lâm Xuân Vũ', phone: '0987654321' },
  { id: 4, name: 'Đới Trường Sinh', phone: '0901235557' },
  { id: 5, name: 'Trần Văn Hoa', phone: '091735678' },
  { id: 6, name: 'Lê Thi Màu', phone: '0987099272' },
  { id: 7, name: 'David Cecia', phone: '03738483212' },
  { id: 8, name: 'Phạm Thành LOng', phone: '0187383822' },
  { id: 9, name: 'Nguyễn Thùy DƯơng', phone: '0992729783' },
];


const Calls = () => {
  const [selectedUser, setSelectedUser] = useState(null);
  const [popupType, setPopupType] = useState(null); // 'call', 'message', 'history'

  const openPopup = (user, type) => {
    setSelectedUser(user);
    setPopupType(type);
  };

  const closePopup = () => {
    setSelectedUser(null);
    setPopupType(null);
  };

  const renderPopupContent = () => {
    if (!selectedUser || !popupType) return null;

    switch (popupType) {
      case 'call':
        return (
          <>
            <h2>📞 Calling...</h2>
            <p>You in calling process to <strong>{selectedUser.name}</strong> - {selectedUser.phone}</p>
            <p>(.....)</p>
          </>
        );
      case 'message':
        return (
          <>
            <h2>💬 Send Message</h2>
            <p>to: <strong>{selectedUser.name}</strong> - {selectedUser.phone}</p>
            <textarea placeholder="Enter content..."></textarea>
            <button className="send-btn" onClick={() => alert("Sent!")}>Send</button>
          </>
        );
      case 'history':
        return (
          <>
            <h2>📂 History</h2>
            <p>History of <strong>{selectedUser.name}</strong>:</p>
            <ul>
              <li>01/07/2025 - 09:00 - Arrived calling</li>
              <li>28/06/2025 - 14:20 - Missing calling</li>
              <li>25/06/2025 - 20:45 - Arrived calling</li>
            </ul>
          </>
        );
      default:
        return null;
    }
  };

  return (
    <div className="calls-container">
      <h1>User</h1>
      <table className="calls-table">
        <thead>
          <tr>
            <th>STT</th>
            <th>Name</th>
            <th>Phone</th>
            <th>Act</th>
          </tr>
        </thead>
        <tbody>
          {mockUsers.map((user, index) => (
            <tr key={user.id}>
              <td>{index + 1}</td>
              <td>{user.name}</td>
              <td>{user.phone}</td>
              <td>
                <button onClick={() => openPopup(user, 'call')}>📞</button>{' '}
                <button onClick={() => openPopup(user, 'message')}>💬</button>{' '}
                <button onClick={() => openPopup(user, 'history')}>📂</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* Popup Form */}
      {selectedUser && popupType && (
        <div className="popup-overlay">
          <div className="popup-box">
            <button className="close-btn" onClick={closePopup}>×</button>
            {renderPopupContent()}
          </div>
        </div>
      )}
    </div>
  );
};

export default Calls;
