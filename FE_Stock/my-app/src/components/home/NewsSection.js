import React from 'react';
import '../home/NewsSection.css';


const NewsSection = () => (
    <div className="news-section">
        <div className="main-article">
            <img src="/path/to/boeing-image.jpg" alt="Boeing" />
            <h2>Boeing posts massive loss as big union vote looms</h2>
            <p>Earnings results highlight the precarious nature...</p>
        </div>
        <div className="other-articles">
            <div className="article">
                <h3>Coca-Cola posts Q3 earnings beat...</h3>
                <p>Yahoo Finance - 2 hours ago</p>
            </div>
        </div>
    </div>
);

export default NewsSection;