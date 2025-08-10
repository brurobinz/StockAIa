import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { useParams, Link } from 'react-router-dom';
import styles from './AccountPage.module.css';
import Navbar from '../components/common/Navbar';
import defaultAvatar from '../assets/avatar-default.jpg';

const AccountPage = () => {
    const { id } = useParams();
    const [userData, setUserData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [avatarTimestamp, setAvatarTimestamp] = useState(new Date().getTime());
    const [countryCode, setCountryCode] = useState('');
    const [localPhoneNumber, setLocalPhoneNumber] = useState('');
    const [depositAmount, setDepositAmount] = useState('');

    const [countries, setCountries] = useState([]);

    const fetchCountries = () => {
        axios.get('https://restcountries.com/v3.1/all?fields=name,cca2,idd')

            .then(response => {
                const countryList = response.data.map(country => ({
                    name: country.name.common,
                    code: country.cca2,
                    callingCode: country.idd?.root + (country.idd?.suffixes ? country.idd.suffixes[0] : ''),
                }));
                countryList.sort((a, b) => a.name.localeCompare(b.name));
                setCountries(countryList);

                // Sau khi danh sách các quốc gia được tải, gọi fetchUserData để lấy thông tin người dùng
                fetchUserData();
            })
            .catch(error => {
                console.error('Error fetching countries data:', error);
                setError('Failed to load country list');
            });
    };
    const handleDeposit = () => {
        const amount = parseFloat(depositAmount);
        if (!amount || amount <= 0) {
            alert('Please enter a valid amount');
            return;
        }

        axios.put(`http://localhost:4000/api/users/${id}/deposit`, { amount }, {
            headers: {
                Authorization: `Bearer ${localStorage.getItem('token')}`
            }
        })
        .then((response) => {
            setUserData(prev => ({
                ...prev,
                walletBalance: response.data.walletBalance,
                walletHistory: response.data.walletHistory
            }));
            setDepositAmount('');
            alert('💰 Deposit successful!');
        })
        .catch((err) => {
            alert('Deposit failed');
            console.error(err);
        });
    };


    const fetchUserData = () => {
        if (!id) {
            setError('No user ID provided.');
            setLoading(false);
            return;
        }

        axios.get(`http://localhost:4000/api/users/${id}`)
            .then(response => {
                setUserData(response.data);
                setLoading(false);

                // Sau khi dữ liệu người dùng được cập nhật, tách số điện thoại
                if (response.data.country && response.data.phoneNumber) {
                    const selectedCountry = countries.find(country => country.name === response.data.country);
                    if (selectedCountry && selectedCountry.callingCode) {
                        const code = selectedCountry.callingCode;
                        setCountryCode(code);
                        // Hiển thị số điện thoại như đã lưu (chỉ số địa phương)
                        setLocalPhoneNumber(response.data.phoneNumber);
                    }
                }
            })
            .catch(error => {
                console.error('Error fetching user data:', error);
                setError('Failed to load user data');
                setLoading(false);
            });
    };

    useEffect(() => {
        console.log("Fetching countries data...");
        fetchCountries();
    }, [id]);    

    useEffect(() => {
        if (userData?.avatar) {
            setAvatarTimestamp(new Date().getTime());
        }
    }, [userData?.avatar]);

    if (loading) {
        return <div>Loading...</div>;
    }

    if (error) {
        return <div>{error}</div>;
    }

    if (!userData) {
        return <div>User not found</div>;
    }

    return (
        <div className={styles.accountPage}>
            <Navbar />
            <nav className={styles.navbar}>
                <a href="#personal-info" className={styles.navLink}>Personal information</a>
                <a href="#security" className={styles.navLink}>Security</a>
                <a href="#recent-activity" className={styles.navLink}>Recent activity</a>
                <a href="#wallet" className={styles.navLink}>Wallet</a>
            </nav>
            <main className={styles.mainContent}>
                <section id="personal-info" className={styles.section}>
                    <h2>Personal Information</h2>
                    <div className={styles.infoContainer}>
                        <div className={styles.details}>
                            <h3>Your details</h3>
                            <p>First name: {userData.firstName || 'N/A'}</p>
                            <p>Last name: {userData.lastName || 'N/A'}</p>
                            <p>Preferred name: {userData.preferredName || 'N/A'}</p>
                            <p>Date of birth: {userData.dateOfBirth ? new Date(userData.dateOfBirth).toLocaleDateString() : 'N/A'}</p>
                            <p>Gender: {userData.gender || 'N/A'}</p>
                            <p>
                                Phone number: {countryCode ? `${countryCode} ${localPhoneNumber}` : userData.phoneNumber || 'N/A'}
                            </p>
                            <p>Country: {userData.country || 'N/A'}</p>
                            <Link to={`/update-personal-details/${id}`} className={styles.updateLink}>Update personal details</Link>
                        </div>
                        <div className={styles.profileSettings}>
                            <div className={styles.userInfo}>
                                <img src={userData.avatar ? `${userData.avatar}?timestamp=${avatarTimestamp}` : defaultAvatar} alt="User Avatar" className={styles.avatarLarge} />
                                <p className={styles.username}>{userData.username}</p> {/* Thêm class để in đậm */}
                                <p>{userData.email}</p>
                                <Link to={`/account/${id}/update-avatar`} className={styles.updateLink}>Update profile avatar</Link>
                            </div>
                            <div className={styles.language}>
                                <h3>Language</h3>
                                <p>English - United Kingdom</p>
                                <a href="/update-language" className={styles.updateLink}>Update language</a>
                            </div>
                        </div>
                    </div>
                </section>
                <section id="security" className={styles.section}>
                    <h2>Security</h2>
                    <div className={styles.infoContainer}>
                        <h3>Login and Security Settings</h3>
                        <p>Email: {userData.email}</p>
                        <p>Password: **********</p>
                        <Link to={`/update-security/${id}`} className={styles.updateLink}>Update password</Link>
                    </div>
                </section>
                <section id="recent-activity" className={styles.section}>
                    <h2>Recent Activity</h2>
                    <div className={styles.activityList}>
                        <div className={styles.activityItem}>
                            <p><strong>✔️ You updated your password</strong></p>
                            <p className={styles.activityTime}>2 minutes ago</p>
                        </div>
                        <div className={styles.activityItem}>
                            <p><strong>📤 You uploaded a new profile avatar</strong></p>
                            <p className={styles.activityTime}>1 hour ago</p>
                        </div>
                        <div className={styles.activityItem}>
                            <p><strong>🔓 You logged in</strong></p>
                            <p className={styles.activityTime}>Today at 09:15 AM</p>
                        </div>
                    </div>
                </section>
                <section id="wallet" className={styles.section}>
                    <h2>Wallet</h2>
                    <div className={styles.walletContainer}>
                        <div className={styles.walletInfo}>
                            <p><strong>Current Balance:</strong> ${userData.walletBalance?.toFixed(2) || '0.00'}</p>
                            <p><strong>Status:</strong> {userData.walletStatus || 'Active'}</p>
                        </div>

                        <div className={styles.walletActions}>
                            <input
                                type="number"
                                placeholder="Enter amount to deposit"
                                className={styles.input}
                                value={depositAmount}
                                onChange={(e) => setDepositAmount(e.target.value)}
                            />
                            <button onClick={handleDeposit} className={styles.button}>Deposit</button>
                        </div>

                        <div className={styles.walletHistory}>
                            <h4>Recent Transactions</h4>
                            <ul>
                                {userData.walletHistory?.length ? (
                                    userData.walletHistory.map((tx, index) => (
                                        <li key={index}>
                                            {tx.date}: {tx.type} ${tx.amount.toFixed(2)}
                                        </li>
                                    ))
                                ) : (
                                    <li>No transactions</li>
                                )}
                            </ul>
                        </div>
                    </div>
                </section>



            </main>
        </div>
    );
};

export default AccountPage;