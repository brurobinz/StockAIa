import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import axios from 'axios';
import logo from '../../assets/logo.png';
import settingsIcon from '../../assets/settings.png';
import privacyIcon from '../../assets/privacy.png';
import helpIcon from '../../assets/help.png';
import addSwitchIcon from '../../assets/add-switch.png';
import defaultAvatar from '../../assets/avatar-default.jpg';

const Header = () => {
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    const [isDropdownOpen, setIsDropdownOpen] = useState(false);
    const [userId, setUserId] = useState(null);
    const [userName, setUserName] = useState('User');
    const [userEmail, setUserEmail] = useState('user@example.com');
    const [searchQuery, setSearchQuery] = useState('');
    const navigate = useNavigate();
    const [userAvatar, setUserAvatar] = useState(defaultAvatar);
    const [avatarTimestamp, setAvatarTimestamp] = useState(new Date().getTime());

    useEffect(() => {
        const token = localStorage.getItem('token');
        const storedUserId = localStorage.getItem('userId');

        if (token && storedUserId) {
            setIsLoggedIn(true);
            setUserId(storedUserId);

            // Fetch user data
            axios
                .get(`http://localhost:4000/api/users/${storedUserId}`)
                .then((response) => {
                    setUserName(response.data.username || 'User');
                    setUserEmail(response.data.email || 'user@example.com');

                    // Kiểm tra avatar, nếu không có thì sử dụng avatar mặc định
                    if (response.data.avatar) {
                        setUserAvatar(response.data.avatar);
                    } else {
                        setUserAvatar(defaultAvatar);
                    }
                })
                .catch((error) => {
                    console.error('Error fetching user data:', error);
                    setUserAvatar(defaultAvatar); // Dùng avatar mặc định khi lỗi
                });
        }
    }, []);

    const handleLogout = () => {
        localStorage.removeItem('token');
        localStorage.removeItem('userId'); // Remove userId as well
        setIsLoggedIn(false);
        setIsDropdownOpen(false);
        navigate('/login'); // Redirect to login page after logout
    };

    const toggleDropdown = () => {
        setIsDropdownOpen(!isDropdownOpen);
    };

    const handleSearchInputChange = (e) => {
        setSearchQuery(e.target.value);
    };

    const handleSearchKeyPress = (e) => {
        if (e.key === 'Enter' && searchQuery.trim() !== '') {
            localStorage.setItem('searchQuery', searchQuery);
            navigate('/ai-consulting');
        }
    };

    return (
        <header className="header">
            <Link to="/">
                <img src={logo} alt="Stock Insight Logo" className="logo" />
            </Link>
            <div className="search-container">
                <input
                    type="text"
                    placeholder="Search for news, symbols or companies..."
                    className="search-bar"
                    value={searchQuery}
                    onChange={handleSearchInputChange}
                    onKeyPress={handleSearchKeyPress}
                />
                <i className="fas fa-search search-icon"></i>
            </div>
            <div className="auth-links">
                {isLoggedIn ? (
                    <div className="user-menu">
                        <img
                            src={`${userAvatar}?timestamp=${avatarTimestamp}`}
                            alt="User Avatar"
                            className="user-avatar"
                            onClick={toggleDropdown}
                        />
                        {isDropdownOpen && (
                            <div className="profileMenu">
                                <p className="profileName">{userName}</p>
                                <p className="profileEmail">{userEmail}</p>
                                <Link to={`/account/${userId}`} className="manageAccount">
                                    Manage Your Account
                                </Link>
                                <div className="profileMenuOptions">
                                    <Link to="/settings" className="profileMenuItem">
                                        <img src={settingsIcon} alt="Settings Icon" className="menuIcon" /> Settings
                                    </Link>
                                    <Link to="/privacy" className="profileMenuItem">
                                        <img src={privacyIcon} alt="Privacy Icon" className="menuIcon" /> Privacy
                                    </Link>
                                    <Link to="/help" className="profileMenuItem">
                                        <img src={helpIcon} alt="Help Icon" className="menuIcon" /> Help
                                    </Link>
                                    <Link to="/switch-account" className="profileMenuItem">
                                        <img src={addSwitchIcon} alt="Add or Switch Accounts Icon" className="menuIcon" /> Add or switch accounts
                                    </Link>
                                </div>
                                <button onClick={handleLogout} className="signOut">
                                    Sign out
                                </button>
                            </div>
                        )}
                    </div>
                ) : (
                    <>
                        <Link to="/login">Log in</Link>
                        <Link to="/register">Register</Link>
                    </>
                )}
            </div>
        </header>
    );
};

export default Header;