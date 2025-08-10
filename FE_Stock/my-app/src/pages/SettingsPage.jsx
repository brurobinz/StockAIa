import React, { useState, useEffect } from 'react';
import './SettingsPage.css';

const SettingsPage = () => {
    const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'light');
    const [language, setLanguage] = useState('English');
    const [notifications, setNotifications] = useState(true);
    const [helpMessage, setHelpMessage] = useState('');
    const [submitted, setSubmitted] = useState(false);

    useEffect(() => {
        // Cập nhật thuộc tính data-theme trên <html>
        if (theme === 'system') {
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            document.documentElement.setAttribute('data-theme', prefersDark ? 'dark' : 'light');
        } else {
            document.documentElement.setAttribute('data-theme', theme);
        }
        localStorage.setItem('theme', theme);
    }, [theme]);

    const handleThemeChange = (e) => setTheme(e.target.value);
    const handleLanguageChange = (e) => setLanguage(e.target.value);
    const handleToggleNotifications = () => setNotifications(!notifications);
    const handleHelpSubmit = () => {
        setSubmitted(true);
        setTimeout(() => setSubmitted(false), 3000);
        setHelpMessage('');
    };

    return (
        <div className="settings-page">
            <h2>⚙️ Settings</h2>

            <div className="setting-section">
                <h3>🎨 Theme</h3>
                <select value={theme} onChange={handleThemeChange}>
                    <option value="light">Light</option>
                    <option value="dark">Dark</option>
                    <option value="system">System Default</option>
                </select>
            </div>

            <div className="setting-section">
                <h3>🔔 Notifications</h3>
                <button onClick={handleToggleNotifications}>
                    {notifications ? 'Disable mock stock alerts' : 'Enable mock stock alerts'}
                </button>
                {notifications && (
                    <div className="mock-notification">
                        📈 Apple stock has increased 2.5% today!
                    </div>
                    
                )}
                {notifications && (
                    <div className="mock-notification">
                        📈 You earn $13.00 from Google Stock Dividend!
                    </div>
                    
                )}
                {notifications && (
                    <div className="mock-notification">
                        🌐 Tesla stock has decreased 1.77% today!
                    </div>
                    
                )}
            </div>

            <div className="setting-section">
                <h3>🌐 Language</h3>
                <select value={language} onChange={handleLanguageChange}>
                    <option>English</option>
                    <option>Vietnamese</option>
                    <option>Korean</option>
                    <option>Japanese</option>
                    <option>Chinese</option>
                    <option>Spanish</option>
                    <option>German</option>
                </select>
            </div>

            <div className="setting-section">
                <h3>📩 Send Help Request to Admin</h3>
                <textarea
                    rows="4"
                    placeholder="Describe your issue or feedback..."
                    value={helpMessage}
                    onChange={(e) => setHelpMessage(e.target.value)}
                />
                <button onClick={handleHelpSubmit}>Send</button>
                {submitted && <p className="success-msg">Your message has been sent to admin ✅</p>}
            </div>
        </div>
    );
};

export default SettingsPage;
