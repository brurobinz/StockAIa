import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import logo from '../../assets/logo.png';
import usernameIcon from '../../assets/information.png';
import emailIcon from '../../assets/email.png';
import passwordIcon from '../../assets/password.png';
import repeatPasswordIcon from '../../assets/repeat-password.png';
import '../signup/SignupForm.css';

const SignupForm = () => {
    const [username, setUsername] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [repeatPassword, setRepeatPassword] = useState('');
    const [month, setMonth] = useState('');
    const [day, setDay] = useState('');
    const [year, setYear] = useState('');
    const [gender, setGender] = useState('');
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    const navigate = useNavigate();

    const months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
    const days = Array.from({ length: 31 }, (_, i) => i + 1);
    const currentYear = new Date().getFullYear();
    const years = Array.from({ length: currentYear - 1900 + 1 }, (_, i) => 1900 + i);

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (password !== repeatPassword) {
            setError("Passwords do not match");
            setSuccess('');
            return;
        }

        if (!gender) {
            setError("Please select your gender");
            setSuccess('');
            return;
        }

        // Create dateOfBirth from day, month, and year fields
        const selectedMonthIndex = months.indexOf(month) + 1;
        const formattedMonth = selectedMonthIndex < 10 ? `0${selectedMonthIndex}` : selectedMonthIndex;
        const formattedDay = day < 10 ? `0${day}` : day;
        const dateOfBirth = `${year}-${formattedMonth}-${formattedDay}`;

        try {
            const response = await fetch(`${process.env.REACT_APP_API_URL}/users/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, email, password, dateOfBirth, gender }), // Send gender
            });

            const data = await response.json();
            if (response.ok) {
                setSuccess("Account created successfully!");
                setError('');
                // Save token and userId to localStorage
                localStorage.setItem('token', data.token);
                localStorage.setItem('userId', data.userId);
                // Redirect to account page or home page
                navigate(`/account/${data.userId}`); // Or navigate('/') to go to the home page
            } else {
                setError(data.message || 'Registration failed');
                setSuccess('');
            }
        } catch (err) {
            console.error('Error during registration:', err);
            setError('An error occurred. Please try again later.');
            setSuccess('');
        }
    };

    return (
        <div>
            <div className="top-links">
                <a href="/help">Help</a>
                <a href="/terms">Terms</a>
                <a href="/privacy">Privacy</a>
            </div>
            <div className="signup-form">
                <img src={logo} alt="Stock Insight Logo" className="form-logo" />
                <h2>Create a new Stock Insight account</h2>
                <form onSubmit={handleSubmit}>
                    <div className="input-container">
                        <img src={usernameIcon} alt="Username Icon" className="input-icon" />
                        <input
                            type="text"
                            placeholder="Username"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            required
                        />
                    </div>
                    <div className="input-container">
                        <img src={emailIcon} alt="Email Icon" className="input-icon" />
                        <input
                            type="email"
                            placeholder="Email"
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
                    <div className="input-container">
                        <img src={repeatPasswordIcon} alt="Repeat Password Icon" className="input-icon" />
                        <input
                            type="password"
                            placeholder="Repeat your password"
                            value={repeatPassword}
                            onChange={(e) => setRepeatPassword(e.target.value)}
                            required
                        />
                    </div>
                    <div className="gender-container">
                        <label>Gender</label>
                        <div className="gender-options">
                            <label>
                                <input
                                    type="radio"
                                    name="gender"
                                    value="Male"
                                    checked={gender === 'Male'}
                                    onChange={(e) => setGender(e.target.value)}
                                    required
                                />
                                Male
                            </label>
                            <label>
                                <input
                                    type="radio"
                                    name="gender"
                                    value="Female"
                                    checked={gender === 'Female'}
                                    onChange={(e) => setGender(e.target.value)}
                                    required
                                />
                                Female
                            </label>
                            <label>
                                <input
                                    type="radio"
                                    name="gender"
                                    value="Other"
                                    checked={gender === 'Other'}
                                    onChange={(e) => setGender(e.target.value)}
                                    required
                                />
                                Other
                            </label>
                        </div>
                    </div>
                    <div className="dob-container">
                        <label>Date of birth</label>
                        <select name="month" required onChange={(e) => setMonth(e.target.value)} value={month}>
                            <option value="">Month</option>
                            {months.map((month, index) => (
                                <option key={index} value={month}>{month}</option>
                            ))}
                        </select>
                        <select name="day" required onChange={(e) => setDay(e.target.value)} value={day}>
                            <option value="">Day</option>
                            {days.map((day) => (
                                <option key={day} value={day}>{day}</option>
                            ))}
                        </select>
                        <select name="year" required onChange={(e) => setYear(e.target.value)} value={year}>
                            <option value="">Year</option>
                            {years.map((year) => (
                                <option key={year} value={year}>{year}</option>
                            ))}
                        </select>
                    </div>
                    {error && <p className="error-message">{error}</p>}
                    {success && <p className="success-message">{success}</p>}
                    <p className="terms">
                        By continuing, you agree to our <a href="/terms">Terms</a> and <a href="/privacy">Privacy Policy</a>.
                    </p>
                    <button type="submit">Next</button>
                </form>
                <p className="login-link">Already have an account? <a href="/login">Log in</a></p>
            </div>
        </div>
    );
};

export default SignupForm;