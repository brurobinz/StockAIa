// App.js
import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import UserList from './pages/UserList';
import Login from './pages/Login';
import UserDetails from './pages/UserDetails';
import Header from './components/Header';
import './App.css';
import Calls from './pages/Calls';
import AdminHelpPage from './pages/AdminHelpPage';
function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(Boolean(localStorage.getItem('token')));

  // Check and update the login status when there is a change from localStorage
  useEffect(() => {
    const handleStorageChange = () => {
      setIsAuthenticated(Boolean(localStorage.getItem('token')));
    };

    // Listen for change events in localStorage
    window.addEventListener('storage', handleStorageChange);

    return () => {
      window.removeEventListener('storage', handleStorageChange);
    };
  }, []);

  // Function to update the login status after a successful login
  const handleLoginSuccess = () => {
    setIsAuthenticated(true);
  };
  console.log('Checkpoint: isAuthenticated log is running');
  console.log('isAuthenticated:', isAuthenticated); // <-- Thêm dòng này

  return (
    <Router>
      <div className="app-container">
        {isAuthenticated && <Sidebar />}
        <div className="main-content">
          <Header />
          <Routes>
  {/* Route login */}
  <Route path="/login" element={<Login onLoginSuccess={handleLoginSuccess} />} />

  {/* Route yêu cầu đăng nhập */}
  <Route
    path="/dashboard"
    element={isAuthenticated ? <Dashboard /> : <Navigate to="/login" />}
  />
  <Route
    path="/users"
    element={isAuthenticated ? <UserList /> : <Navigate to="/login" />}
  />
  <Route
    path="/users/:userId"
    element={isAuthenticated ? <UserDetails /> : <Navigate to="/login" />}
  />
  <Route path="/calls" element={isAuthenticated ? <Calls /> : <Navigate to="/login" />} />
  <Route path="/papers" element={<AdminHelpPage />} />
</Routes> 
        </div>
      </div>
    </Router>
  );
}

export default App;