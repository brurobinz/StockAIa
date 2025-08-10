import React from 'react';
import '../home/MarketSummary.css';

const MarketSummary = () => (
    <aside className="market-summary">
        <h3>Markets</h3>
        <ul>
            <li>S&P 500 <span className="red">-28.81 (-0.68%)</span></li>
            <li>Nasdaq <span className="red">-122.12 (-0.68%)</span></li>
        </ul>
    </aside>
);

export default MarketSummary;
