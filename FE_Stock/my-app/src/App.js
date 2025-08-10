import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './App.css';
import HomePage from './pages/HomePage';
import SignUpPage from './pages/SignUpPage';
import LoginPage from './pages/LoginPage';
import AIConsultingPage from './pages/AIConsultingPage';
import AccountPage from './pages/AccountPage';
import UpdatePersonalDetailsPage from './pages/UpdatePersonalDetailsPage';
import UpdateAvatarPage from './pages/UpdateAvatarPage';
import ArticleDetailPage from './pages/ArticleDetailPage';
import '@fortawesome/fontawesome-free/css/all.min.css';
import AnalysisPage from './pages/AnalysisPage';
import MarketPage from './pages/MarketPage';
import UpdateSecurityPage from './pages/UpdateSecurityPage';
import SettingsPage from './pages/SettingsPage';
import UserHelpPage from './pages/UserHelpPage';
import PrivacyPage from './pages/PrivacyPage';
function App() {
    useEffect(() => {
        const updateVisitCount = async () => {
            try {
                await fetch('http://localhost:4000/api/users/increment-visit', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                });
            } catch (error) {
                console.error('Error updating visit count:', error);
            }
        };

        updateVisitCount();
    }, []);

    return (
        <Router>
            <Routes>
                <Route path="/" element={<HomePage />} />
                <Route path="/register" element={<SignUpPage />} />
                <Route path="/login" element={<LoginPage />} />
                <Route path="/ai-consulting" element={<AIConsultingPage />} />
                <Route path="/account/:id" element={<AccountPage />} />
                <Route path="/update-personal-details/:id" element={<UpdatePersonalDetailsPage />} />
                <Route path="/account/:id/update-avatar" element={<UpdateAvatarPage />} />
                <Route path="/article/:articleId" element={<ArticleDetailPage />} />
                <Route path="/analysis" element={<AnalysisPage />} />
                <Route path="/markets" element={<MarketPage />} />
                <Route path="/update-security/:id" element={<UpdateSecurityPage />} />
                <Route path="/settings" element={<SettingsPage />} />
                <Route path="/help" element={<UserHelpPage />} />
                <Route path="/privacy" element={<PrivacyPage />} />
            </Routes>
        </Router>
    );
}

export default App;