import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import logo from '../../assets/logo.png';
import emailIcon from '../../assets/email.png';
import passwordIcon from '../../assets/password.png';
import '../login/LoginForm.css';

const LoginForm = () => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();

        try {
            const response = await fetch(`${process.env.REACT_APP_API_URL}/users/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password }),
            });

            const data = await response.json();
            if (response.ok) {
                alert('Login successful');
                localStorage.setItem('token', data.token); // Lưu token vào localStorage
                localStorage.setItem('userId', data.userId); // Lưu userId vào localStorage
                navigate('/'); // Chuyển hướng đến trang homepage
            } else {
                setError(data.message || 'Login failed');
            }
        } catch (err) {
            console.error('Error during login:', err);
            setError('An error occurred. Please try again.');
        }
    };

    return (
        <div className="login-form">
            <img src={logo} alt="Stock Insight Logo" className="form-logo" />
            <h2>Log in to Stock Insight</h2>
            <form onSubmit={handleSubmit}>
                <div className="input-container">
                    <img src={emailIcon} alt="Email Icon" className="input-icon" />
                    <input
                        type="email"
                        placeholder="Your email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        required
                    />
                </div>
                <div className="input-container">
                    <img src={passwordIcon} alt="Password Icon" className="input-icon" />
                    <input
                        type="password"
                        placeholder="Password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        required
                    />
                </div>
                {error && <p className="error-message">{error}</p>}
                <button type="submit">Log in</button>
            </form>
            <p className="forgot-password">
                <a href="/forgot-password">Forgot password?</a>
            </p>
            <p className="register-link">
                Do not have an account? <a href="/register">Register now!</a>
            </p>
        </div>
    );
};

export default LoginForm;
