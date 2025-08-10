import React, { useEffect, useState } from 'react';
import '../common/Navbar.css';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';
import defaultAvatar from '../../assets/avatar-default.jpg';
import settingsIcon from '../../assets/settings.png';
import privacyIcon from '../../assets/privacy.png';
import helpIcon from '../../assets/help.png';
import addSwitchIcon from '../../assets/add-switch.png';

const Navbar = () => {
    const location = useLocation();
    const navigate = useNavigate();

    const [isLoggedIn, setIsLoggedIn] = useState(false);
    const [isDropdownOpen, setIsDropdownOpen] = useState(false);
    const [userName, setUserName] = useState('User');
    const [userEmail, setUserEmail] = useState('user@example.com');
    const [userAvatar, setUserAvatar] = useState(defaultAvatar);
    const [userId, setUserId] = useState(null);

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
                    setUserAvatar(response.data.avatar || defaultAvatar);
                })
                .catch((error) => {
                    console.error('Error fetching user data:', error);
                    setUserAvatar(defaultAvatar);
                });
        }
    }, []);

    const toggleDropdown = () => {
        setIsDropdownOpen(!isDropdownOpen);
    };

    const handleLogout = () => {
        localStorage.removeItem('token');
        localStorage.removeItem('userId');
        setIsLoggedIn(false);
        setIsDropdownOpen(false);
        navigate('/login');
    };

    return (
        <nav className="navbar">
            {/* Logo */}
            <div className="navbar-brand">
                <Link to="/">
                    <span className="brand-stock">Stock</span>
                    <span className="brand-insight">Insight</span>
                </Link>
            </div>

            {/* Centered Navbar Links */}
            <ul className="navbar-links">
                <li>
                    <Link
                        to="/"
                        className={location.pathname === '/' ? 'active' : ''}
                    >
                        News
                    </Link>
                </li>
                <li>
                    <Link
                        to="/markets"
                        className={location.pathname === '/markets' ? 'active' : ''}
                    >
                        Markets
                    </Link>
                </li>
                <li>
                    <Link
                        to="/ai-consulting"
                        className={location.pathname === '/ai-consulting' ? 'active' : ''}
                    >
                        AI Consulting
                    </Link>
                </li>
                <li>
                    <Link
                        to="/analysis"
                        className={location.pathname === '/analysis' ? 'active' : ''}
                    >
                        Analysis
                    </Link>
                </li>
            </ul>

            {/* Log in/Register or User Dropdown */}
            <div className="auth-links">
                {isLoggedIn ? (
                    <div className="user-menu">
                        <img
                            src={userAvatar}
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
                                        <img src={addSwitchIcon} alt="Switch Accounts Icon" className="menuIcon" /> Add or Switch Accounts
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
                        <Link to="/login" className="auth-link">Log in</Link>
                        <Link to="/register" className="auth-link">Register</Link>
                    </>
                )}
            </div>
        </nav>
    );
};

export default Navbar;