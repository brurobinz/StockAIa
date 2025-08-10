// src/pages/Dashboard.js
import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import Header from '../components/Header';
import StockTable from '../components/StockTable';
import DashboardCard from '../components/DashboardCard';
import axios from 'axios';
import '../pages/Dashboard.css';
import useVisit from '../hooks/useVisit';

const Dashboard = ({ isAdmin }) => {
    const [stocks, setStocks] = useState([]);
    const [loading, setLoading] = useState(true);
    const [updating, setUpdating] = useState(false);
    const [expandedRows, setExpandedRows] = useState([]);
    const [chartData, setChartData] = useState({});
    const [loadingChart, setLoadingChart] = useState(false);
    const isFetchingRef = useRef(false);
    const [userCount, setUserCount] = useState(0);

    const visitCount = useVisit();

    const stockSymbols = useMemo(() => [
        'NVDA', 'INTC', 'PLTR', 'TSLA', 'AAPL', 'BBD', 'T', 'SOFI',
        'WBD', 'SNAP', 'NIO', 'BTG', 'F', 'AAL', 'NOK', 'BAC',
        'CCL', 'ORCL', 'AMD', 'PFE', 'KGC', 'MARA', 'SLB', 'NU',
        'MPW', 'MU', 'LCID', 'NCLH', 'RIG', 'AMZN', 'ABEV', 'U',
        'LUMN', 'AGNC', 'VZ', 'WBA', 'WFC', 'RIVN', 'UPST', 'CFE',
        'CSCO', 'VALE', 'AVGO', 'PBR', 'GOOGL', 'SMMT', 'GOLD',
        'NGC', 'BCS', 'UAA'
    ], []);

    const RAPIDAPI_KEY = process.env.REACT_APP_RAPIDAPI_KEY;

    if (!RAPIDAPI_KEY) {
        console.error('REACT_APP_RAPIDAPI_KEY is not set. Please set it in your .env file.');
    }

    const saveStocksToLocalStorage = (stocksData) => {
        try {
            const serializedData = JSON.stringify(stocksData);
            localStorage.setItem('stocksData', serializedData);
        } catch (error) {
            console.error('Error saving to localStorage:', error);
        }
    };

    const loadStocksFromLocalStorage = () => {
        try {
            const serializedData = localStorage.getItem('stocksData');
            if (serializedData === null) return [];
            return JSON.parse(serializedData);
        } catch (error) {
            console.error('Error loading from localStorage:', error);
            return [];
        }
    };

    const fetchStockData = async (symbol) => {
        if (!RAPIDAPI_KEY) return;
        const options = {
            method: 'GET',
            url: `https://realstonks.p.rapidapi.com/stocks/${symbol}/advanced`,
            headers: {
                'x-rapidapi-host': 'realstonks.p.rapidapi.com',
                'x-rapidapi-key': RAPIDAPI_KEY,
            },
        };

        try {
            const response = await axios.request(options);
            const data = response.data;

            if (data) {
                setStocks(prevStocks => {
                    const isExisting = prevStocks.some(stock => stock.symbol === symbol);
                    if (isExisting) {
                        const updatedStocks = prevStocks.map(stock =>
                            stock.symbol === symbol
                                ? {
                                    ...stock,
                                    companyName: data.symbolName || stock.companyName,
                                    price: data.lastPrice !== undefined ? data.lastPrice : stock.price,
                                    change: data.percentChange !== undefined ? (data.percentChange * 100) : stock.change,
                                    volume: data.volume !== undefined ? data.volume.toLocaleString() : stock.volume,
                                    priceChange: data.priceChange || stock.priceChange,
                                    lastPrice: data.lastPrice,
                                    percentChange: data.percentChange,
                                    bidPrice: data.bidPrice,
                                    askPrice: data.askPrice,
                                    bidSize: data.bidSize,
                                    askSize: data.askSize,
                                    tradeTime: data.tradeTime,
                                    lowPrice: data.lowPrice,
                                    openPrice: data.openPrice,
                                    highPrice: data.highPrice,
                                    previousPrice: data.previousPrice,
                                    averageVolume: data.averageVolume,
                                    stochasticK14d: data.stochasticK14d,
                                    weightedAlpha: data.weightedAlpha,
                                    priceChange5d: data.priceChange5d,
                                    percentChange5d: data.percentChange5d,
                                    lowPrice1y: data.lowPrice1y,
                                    highPrice1y: data.highPrice1y,
                                }
                                : stock
                        );
                        saveStocksToLocalStorage(updatedStocks);
                        return updatedStocks;
                    }

                    const newStock = {
                        symbol: data.symbol || symbol,
                        companyName: data.symbolName || symbol,
                        price: data.lastPrice !== undefined ? data.lastPrice : 'N/A',
                        change: data.percentChange !== undefined ? (data.percentChange * 100) : 0,
                        volume: data.volume !== undefined ? data.volume.toLocaleString() : '0',
                        priceChange: data.priceChange || 'N/A',
                        lastPrice: data.lastPrice,
                        percentChange: data.percentChange,
                        bidPrice: data.bidPrice,
                        askPrice: data.askPrice,
                        bidSize: data.bidSize,
                        askSize: data.askSize,
                        tradeTime: data.tradeTime,
                        lowPrice: data.lowPrice,
                        openPrice: data.openPrice,
                        highPrice: data.highPrice,
                        previousPrice: data.previousPrice,
                        averageVolume: data.averageVolume,
                        stochasticK14d: data.stochasticK14d,
                        weightedAlpha: data.weightedAlpha,
                        priceChange5d: data.priceChange5d,
                        percentChange5d: data.percentChange5d,
                        lowPrice1y: data.lowPrice1y,
                        highPrice1y: data.highPrice1y,
                    };
                    const updatedStocks = [...prevStocks, newStock];
                    saveStocksToLocalStorage(updatedStocks);
                    return updatedStocks;
                });
            } else {
                console.error(`No data found for stock ${symbol}.`);
            }
        } catch (error) {
            console.error(`Error fetching stock data for ${symbol}:`, error);
        }
    };

    const fetchAllStockData = useCallback(async () => {
        if (!RAPIDAPI_KEY) return;
        if (isFetchingRef.current) {
            console.warn('Fetch operation already in progress.');
            return;
        }

        isFetchingRef.current = true;
        setUpdating(true);

        try {
            for (let i = 0; i < stockSymbols.length; i++) {
                const symbol = stockSymbols[i];
                await fetchStockData(symbol);
                await new Promise(resolve => setTimeout(resolve, 1500));
            }
        } catch (error) {
            console.error('Error in fetchAllStockData:', error);
        } finally {
            isFetchingRef.current = false;
            setLoading(false);
            setUpdating(false);
        }
    }, [stockSymbols, RAPIDAPI_KEY]);

    useEffect(() => {
        const savedStocks = loadStocksFromLocalStorage();
        if (savedStocks.length > 0) {
            setStocks(savedStocks);
            setLoading(false);
        }

        fetchAllStockData();

        const interval = setInterval(() => {
            fetchAllStockData();
        }, 15 * 60 * 1000);

        return () => clearInterval(interval);
    }, [fetchAllStockData]);

    useEffect(() => {
        const fetchUserCount = async () => {
            const token = localStorage.getItem('token');
            if (!token) {
                console.error('No token found');
                return;
            }

            try {
                const response = await axios.get('http://localhost:4000/api/users/count', {
                    headers: { Authorization: `Bearer ${token}` },
                });
                console.log('User count response:', response.data);
                setUserCount(response.data.count);
            } catch (error) {
                console.error('Error fetching user count:', error.response?.data || error.message);
            }
        };

        fetchUserCount();
    }, []);

    const handleRowClick = async (symbol) => {
        const isExpanded = expandedRows.includes(symbol);
        if (isExpanded) {
            setExpandedRows(expandedRows.filter(item => item !== symbol));
        } else {
            setExpandedRows([...expandedRows, symbol]);
            if (!chartData[symbol]) {
                await fetchChartData(symbol);
            }
        }
    };

    const fetchChartData = async (symbol) => {
        if (!RAPIDAPI_KEY) return;
        setLoadingChart(true);
        const fixedRange = '1d';
        const options = {
            method: 'GET',
            url: 'https://yahoo-finance166.p.rapidapi.com/api/stock/get-chart',
            params: {
                region: 'US',
                range: fixedRange,
                symbol: symbol,
                interval: '5m',
            },
            headers: {
                'x-rapidapi-host': 'yahoo-finance166.p.rapidapi.com',
                'x-rapidapi-key': RAPIDAPI_KEY,
            },
        };

        try {
            const response = await axios.request(options);
            const data = response.data;

            if (data && data.chart && data.chart.result && data.chart.result.length > 0) {
                const timestamps = data.chart.result[0].timestamp;
                const closes = data.chart.result[0].indicators.quote[0].close;

                const formattedTimestamps = timestamps.map(ts => {
                    const date = new Date(ts * 1000);
                    return `${date.getHours()}:${date.getMinutes().toString().padStart(2, '0')}`;
                });

                const chartConfig = {
                    labels: formattedTimestamps,
                    datasets: [
                        {
                            label: `${symbol} Price`,
                            data: closes,
                            fill: true,
                            backgroundColor: 'rgba(26, 115, 232, 0.2)',
                            borderColor: '#1A73E8',
                            borderWidth: 2,
                            pointRadius: 0,
                            tension: 0.4,
                            pointHoverRadius: 5,
                            pointBackgroundColor: '#1A73E8',
                            pointBorderColor: '#ffffff',
                            pointHoverBackgroundColor: '#ffffff',
                            pointHoverBorderColor: '#1A73E8',
                            cubicInterpolationMode: 'monotone',
                        },
                    ],
                };

                const chartOptions = {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false,
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false,
                            backgroundColor: 'rgba(0,0,0,0.7)',
                            titleColor: '#ffffff',
                            bodyColor: '#ffffff',
                            borderColor: '#1A73E8',
                            borderWidth: 1,
                            callbacks: {
                                label: function (context) {
                                    let label = context.dataset.label || '';
                                    if (label) {
                                        label += ': ';
                                    }
                                    if (context.parsed.y !== null) {
                                        label += `$${context.parsed.y.toFixed(2)}`;
                                    }
                                    return label;
                                }
                            }
                        },
                    },
                    interaction: {
                        mode: 'nearest',
                        axis: 'x',
                        intersect: false,
                    },
                    scales: {
                        x: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Time',
                                color: '#333333',
                                font: {
                                    size: 14,
                                    weight: 'bold',
                                }
                            },
                            grid: {
                                display: false,
                            },
                            ticks: {
                                color: '#333333',
                                font: {
                                    size: 12,
                                }
                            }
                        },
                        y: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Price ($)',
                                color: '#333333',
                                font: {
                                    size: 14,
                                    weight: 'bold',
                                }
                            },
                            grid: {
                                color: 'rgba(0, 0, 0, 0.05)',
                            },
                            ticks: {
                                color: '#333333',
                                font: {
                                    size: 12,
                                },
                                callback: function (value) {
                                    return `$${value}`;
                                }
                            }
                        }
                    }
                };

                setChartData(prevData => ({
                    ...prevData,
                    [symbol]: { data: chartConfig, options: chartOptions },
                }));
            } else {
                console.error(`No chart data found for stock ${symbol}.`);
            }
        } catch (error) {
            console.error(`Error fetching chart data for ${symbol}:`, error);
        } finally {
            setLoadingChart(false);
        }
    };

    return (
        <div className="dashboard">
            <div className="dashboard-content">
                <Header />
                <div className="centered-content">
                    <DashboardCard stockCount={stocks.length} userCount={userCount} visitCount={visitCount} />
                    <div className="stock-table-container">
                        <StockTable
                            stocks={stocks}
                            loading={loading}
                            updating={updating}
                            expandedRows={expandedRows}
                            handleRowClick={handleRowClick}
                            chartData={chartData}
                            loadingChart={loadingChart}
                            isAdmin={isAdmin}
                        />
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
