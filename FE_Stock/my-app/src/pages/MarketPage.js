import React from 'react';
import StockTable from '../components/common/StockTable';
import Navbar from '../components/common/Navbar';
import Footer from '../components/common/Footer';

const MarketPage = () => {
    return (
        <div>
            <Navbar />
            <StockTable />
            <Footer />
        </div>
    );
};

export default MarketPage;
