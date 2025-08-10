import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FaHome, FaUser, FaPhone, FaFileAlt, FaCog, FaUserCircle } from 'react-icons/fa';
import '../components/Sidebar.css';

const Sidebar = () => {
    const navigate = useNavigate();
    const [showDropdown, setShowDropdown] = useState(false);
    const [isLoggedIn, setIsLoggedIn] = useState(false);

    useEffect(() => {
        const token = localStorage.getItem('token');
        setIsLoggedIn(!!token);
    }, []);

    const handleLogout = () => {
        localStorage.removeItem('token');
        setIsLoggedIn(false);
        navigate('/login');
    };

    const handleAvatarClick = () => {
        if (isLoggedIn) {
            setShowDropdown(!showDropdown);
        }
    };

    const handleHomeClick = () => {
        if (isLoggedIn) {
            navigate('/dashboard');
        } else {
            navigate('/login');
        }
    };

    return (
        <div className="sidebar">
            <FaHome className="icon logo" size={30} onClick={handleHomeClick} />
            <FaUser className="icon" size={24} onClick={() => navigate('/users')} />
            <FaPhone className="icon" size={24} onClick={() => navigate('/calls')} />
            <FaFileAlt className="icon" size={24} onClick={() => navigate('/papers')} />
            <FaCog className="icon" size={24} onClick={() => navigate('/settings')} />

            <div className="avatar-container" onClick={handleAvatarClick}>
                <FaUserCircle className="icon avatar" size={24} />
                {isLoggedIn && showDropdown && (
                    <div className="dropdown-menu">
                        <button className="logout-button" onClick={handleLogout}>Log Out</button>
                    </div>
                )}
            </div>
        </div>
    );
};

export default Sidebar;