import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from '../utils/axiosConfig'; // Import axios instance with interceptors
import '../pages/UserList.css';

const UserList = () => {
    const [users, setUsers] = useState([]);
    const navigate = useNavigate();

    useEffect(() => {
        const fetchUsers = async () => {
            const token = localStorage.getItem('token'); // Retrieve token from localStorage
            try {
                const response = await axios.get('http://localhost:4000/api/users', {
                    headers: {
                        'Authorization': `Bearer ${token}` // Include token in request
                    }
                });
                setUsers(response.data); // Update state with user data
            } catch (error) {
                console.error('Error fetching users:', error);
                if (error.response && error.response.status === 401) {
                    alert('Unauthorized access. Please log in as admin.');
                } else if (error.response && error.response.data.msg) {
                    alert(error.response.data.msg);
                } else {
                    alert('Error fetching users.');
                }
            }
        };

        fetchUsers();
    }, []);

    return (
        <div className="user-list-container">
            <div className="main-content">
                <div className="user-list">
                    <h1>Users</h1>
                    <table className="user-table">
                        <thead>
                            <tr>
                                <th>Number</th>
                                <th>Username</th>
                                <th>Email</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {users.map((user, index) => (
                                <tr 
                                    key={user._id} 
                                    onClick={() => navigate(`/users/${user._id}`)} 
                                    style={{ cursor: 'pointer' }}
                                >
                                    <td>{index + 1}</td>
                                    <td>{user.username}</td>
                                    <td>{user.email}</td>
                                    <td>
                                        <span className={`status ${user.status ? user.status.toLowerCase() : ''}`}>
                                            {user.status || 'Unknown'}
                                        </span>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};

export default UserList;