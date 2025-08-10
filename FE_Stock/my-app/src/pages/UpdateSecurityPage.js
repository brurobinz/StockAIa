import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import axios from 'axios';
import styles from './UpdateSecurityPage.module.css';

const generateCaptcha = () => {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    let captcha = '';
    for (let i = 0; i < 5; i++) {
        captcha += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return captcha;
};

const UpdateSecurityPage = () => {
    const { id } = useParams();
    const [oldPassword, setOldPassword] = useState('');
    const [newPassword, setNewPassword] = useState('');
    const [captcha, setCaptcha] = useState('');
    const [inputCaptcha, setInputCaptcha] = useState('');
    const [message, setMessage] = useState('');
    const [error, setError] = useState('');
    const [captchaValue, setCaptchaValue] = useState('');

    useEffect(() => {
        setCaptchaValue(generateCaptcha());
    }, []);

    const handleUpdate = () => {
        const token = localStorage.getItem('token');
        setMessage('');
        setError('');

        if (inputCaptcha !== captchaValue) {
            setError('❌ Incorrect Capcha. Please try again.');
            setCaptchaValue(generateCaptcha());
            return;
        }

        axios.put(
            `http://localhost:4000/api/users/${id}/update-password`,
            { oldPassword, newPassword },
            {
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                }
            }
        )
        .then(() => {
            setMessage('✅ Password updated successfully!');
            setOldPassword('');
            setNewPassword('');
            setInputCaptcha('');
            setCaptchaValue(generateCaptcha());
        })
        .catch((err) => {
            setError('❌ Failed to update password. Please check your old password.');
            setCaptchaValue(generateCaptcha());
        });
    };

    return (
        <div className={styles.updateContainer}>
            <h2 className={styles.title}>Update Password</h2>
            <input
                type="password"
                placeholder="Old password"
                value={oldPassword}
                onChange={(e) => setOldPassword(e.target.value)}
                className={styles.input}
            />
            <input
                type="password"
                placeholder="New password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                className={styles.input}
            />
            <div className={styles.captchaBox}>
                <div className={styles.captchaText}>{captchaValue}</div>
                <input
                    type="text"
                    placeholder="Enter captcha"
                    value={inputCaptcha}
                    onChange={(e) => setInputCaptcha(e.target.value.toUpperCase())}
                    className={styles.input}
                />
            </div>
            <button onClick={handleUpdate} className={styles.button}>
                Update Password
            </button>
            {message && <p className={styles.message}>{message}</p>}
            {error && <p className={`${styles.message} ${styles.error}`}>{error}</p>}
        </div>
    );
};

export default UpdateSecurityPage;
