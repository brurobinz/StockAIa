import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useParams, useNavigate } from 'react-router-dom';
import styles from '../pages/UpdatePersonalDetailsPage.module.css';

const UpdatePersonalDetailsPage = () => {
    const { id } = useParams();
    const navigate = useNavigate();
    const [userData, setUserData] = useState({
        firstName: '',
        lastName: '',
        preferredName: '',
        dateOfBirth: '',
        gender: '',
        phoneNumber: '',
        country: '',
    });
    const [countryCode, setCountryCode] = useState('');
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(true);
    const [countries, setCountries] = useState([]);

    useEffect(() => {
        if (!id) {
            setError('No user ID provided.');
            setLoading(false);
            return;
        }

        axios.get(`http://localhost:4000/api/users/${id}`)
            .then(response => {
                const user = response.data;
                const phoneNumber = user.phoneNumber || '';

                let extractedCountryCode = '';
                let localPhoneNumber = phoneNumber;

                // Assume phoneNumber starts with '+' followed by area code
                if (phoneNumber.startsWith('+')) {
                    const match = phoneNumber.match(/^\+(\d{1,3})/); // Depending on the area code, it can be 1-3 digits
                    if (match) {
                        extractedCountryCode = `+${match[1]}`;
                        localPhoneNumber = phoneNumber.substring(match[0].length);
                    }
                }

                setCountryCode(extractedCountryCode);
                setUserData({ 
                    ...user, 
                    phoneNumber: localPhoneNumber 
                });
                setLoading(false);
            })
            .catch(error => {
                console.error('Error fetching user data:', error);
                setError('Failed to load user data');
                setLoading(false);
            });
    }, [id]);

    useEffect(() => {
        axios.get('https://restcountries.com/v3.1/all?fields=name,cca2,idd')
            .then(response => {
                const countryList = response.data.map(country => ({
                    name: country.name.common,
                    code: country.cca2,
                    callingCode: country.idd?.root + (country.idd?.suffixes ? country.idd.suffixes[0] : ''),
                }));
                countryList.sort((a, b) => a.name.localeCompare(b.name));
                setCountries(countryList);
            })
            .catch(error => {
                console.error('Error fetching countries data:', error);
                setError('Failed to load country list');
            });
    }, []);

    const handleChange = (e) => {
        const { name, value } = e.target;

        if (name === 'country') {
            const selectedCountry = countries.find(country => country.name === value);
            const newCountryCode = selectedCountry ? selectedCountry.callingCode : '';
            setCountryCode(newCountryCode);
            setUserData({ ...userData, country: value, phoneNumber: '' });
        } else if (name === 'phoneNumber') {
            setUserData({ ...userData, phoneNumber: value });
        } else {
            setUserData({ ...userData, [name]: value });
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        try {
            const updatedData = {
                firstName: userData.firstName,
                lastName: userData.lastName,
                preferredName: userData.preferredName,
                dateOfBirth: userData.dateOfBirth,
                gender: userData.gender,
                phoneNumber: countryCode + userData.phoneNumber, // Chỉ thêm countryCode một lần
                country: userData.country,
            };

            const response = await axios.put(`http://localhost:4000/api/users/${id}`, updatedData, {
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('token')}`,
                },
            });

            console.log(response.data); // Confirm the response from the server
            alert('Personal details updated successfully!');
            navigate(`/account/${id}`);
        } catch (error) {
            console.error('Error updating user details:', error);
            setError('Failed to update personal details');
        }
    };       

    if (loading) return <div>Loading...</div>;
    if (error) return <div>{error}</div>;

    return (
        <div className={styles.updatePage}>
            <h2>Update Personal Details</h2>
            <form onSubmit={handleSubmit} className={styles.updateForm}>
                <label>
                    First Name:
                    <input
                        type="text"
                        name="firstName"
                        value={userData.firstName || ''}
                        onChange={handleChange}
                        required
                    />
                </label>
                <label>
                    Last Name:
                    <input
                        type="text"
                        name="lastName"
                        value={userData.lastName || ''}
                        onChange={handleChange}
                        required
                    />
                </label>
                <label>
                    Preferred Name:
                    <input
                        type="text"
                        name="preferredName"
                        value={userData.preferredName || ''}
                        onChange={handleChange}
                    />
                </label>
                <label>
                    Date of Birth:
                    <input
                        type="date"
                        name="dateOfBirth"
                        value={userData.dateOfBirth ? new Date(userData.dateOfBirth).toISOString().split('T')[0] : ''}
                        onChange={handleChange}
                        required
                    />
                </label>
                <label>
                    Gender:
                    <select
                        name="gender"
                        value={userData.gender || ''}
                        onChange={handleChange}
                        required
                    >
                        <option value="" className={styles.defaultOption}>Select Gender</option>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                        <option value="Other">Other</option>
                    </select>
                </label>
                <label>
                    Phone Number:
                    <div className={styles.phoneNumberContainer}>
                        <input
                            type="text"
                            value={countryCode}
                            readOnly
                            className={styles.countryCodeInput}
                        />
                        <input
                            type="tel"
                            name="phoneNumber"
                            value={userData.phoneNumber || ''}
                            onChange={handleChange}
                            required
                            placeholder="Enter phone number"
                            className={styles.phoneNumberInput}
                        />
                    </div>
                </label>
                <label>
                    Country:
                    <select
                        name="country"
                        value={userData.country || ''}
                        onChange={handleChange}
                        required
                    >
                        <option value="" className={styles.defaultOption}>Select Country</option>
                        {countries.map((country) => (
                            <option key={country.code} value={country.name}>
                                {country.name}
                            </option>
                        ))}
                    </select>
                </label>
                <div className={styles.submitButtonContainer}>
                    <button type="submit" className={styles.submitButton}>Save Changes</button>
                </div>
            </form>
        </div>
    );
};

export default UpdatePersonalDetailsPage;