import React from 'react';
import lightIcon from '../assets/light.png';
import DashboardCard from './DashboardCard';
import '../components/Header.css';

const Header = () => {
    return (
        <div className="header">
            <h1>Dashboard</h1>
            <img src={lightIcon} alt="Light Icon" className="light-icon" />
        </div>
    );
};

export default Header;
