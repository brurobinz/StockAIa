// src/hooks/useVisit.js
import { useState, useEffect } from 'react';
import axios from 'axios';

const useVisit = () => {
    const [visitCount, setVisitCount] = useState(0);

    useEffect(() => {
        const token = localStorage.getItem('token'); // Lấy token từ localStorage

        if (!token) {
            console.error('No token found');
            return;
        }

        const incrementVisit = async () => {
            try {
                const response = await axios.post(
                    'http://localhost:4000/api/users/increment-visit',
                    {}, // Body trống
                    {
                        headers: { Authorization: `Bearer ${token}` },
                    }
                );
                setVisitCount(response.data.count);
            } catch (error) {
                console.error('Error incrementing visit:', error.response?.data || error.message);
            }
        };

        const fetchVisitCount = async () => {
            try {
                const response = await axios.get('http://localhost:4000/api/users/visit-count', {
                    headers: { Authorization: `Bearer ${token}` },
                });
                setVisitCount(response.data.count);
            } catch (error) {
                console.error('Error fetching visit count:', error.response?.data || error.message);
            }
        };

        const lastVisit = localStorage.getItem('lastVisit');

        // Chỉ tăng lượt truy cập nếu chưa tăng trong vòng 1 giờ
        const now = new Date().getTime();
        if (!lastVisit || now - lastVisit > 60 * 60 * 1000) {
            incrementVisit();
            localStorage.setItem('lastVisit', now);
        }

        // Luôn luôn lấy lượt truy cập hiện tại
        fetchVisitCount();
    }, []);

    return visitCount;
};

export default useVisit;
