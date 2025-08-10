import React, { useEffect, useState, useCallback, useMemo, useRef } from 'react';
import axios from 'axios';
import './StockTable.css';
import { Line } from 'react-chartjs-2';
import 'chart.js/auto'; // Necessary for Chart.js

const StockTable = () => {
    const [stocks, setStocks] = useState([]);
    const [loading, setLoading] = useState(true);
    const [updating, setUpdating] = useState(false);
    const [expandedRows, setExpandedRows] = useState([]); // Tracks expanded rows
    const [chartData, setChartData] = useState({}); // Stores chart data per stock
    const [loadingChart, setLoadingChart] = useState(false); // Manages chart loading state
    const isFetchingRef = useRef(false); // Prevent overlapping fetches

    // Initialize stock symbols with useMemo
    const stockSymbols = useMemo(() => [
        'NVDA', 'INTC', 'PLTR', 'TSLA', 'AAPL', 'BBD', 'T', 'SOFI',
        'WBD', 'SNAP', 'NIO', 'BTG', 'F', 'AAL', 'NOK', 'BAC',
        'CCL', 'ORCL', 'AMD', 'PFE', 'KGC', 'MARA', 'SLB', 'NU',
        'MPW', 'MU', 'LCID', 'NCLH', 'RIG', 'AMZN', 'ABEV', 'U',
        'LUMN', 'AGNC', 'VZ', 'WBA', 'WFC', 'RIVN', 'UPST', 'CFE',
        'CSCO', 'VALE', 'AVGO', 'PBR', 'GOOGL', 'SMMT', 'GOLD',
        'NGC', 'BCS', 'UAA'
    ], []);

    // API key from environment variables
    const RAPIDAPI_KEY = process.env.REACT_APP_RAPIDAPI_KEY;

    // Function to save stocks to localStorage
    const saveStocksToLocalStorage = (stocksData) => {
        try {
            const serializedData = JSON.stringify(stocksData);
            localStorage.setItem('stocksData', serializedData);
        } catch (error) {
            console.error('Error saving to localStorage:', error);
        }
    };

    // Function to load stocks from localStorage
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

    // Function to fetch stock data from RealStonks API
    const fetchStockData = async (symbol) => {
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
                        // Update existing stock data
                        const updatedStocks = prevStocks.map(stock =>
                            stock.symbol === symbol
                                ? {
                                    ...stock,
                                    companyName: data.symbolName || stock.companyName,
                                    price: data.lastPrice !== undefined ? data.lastPrice : stock.price,
                                    change: data.percentChange !== undefined ? (data.percentChange * 100) : stock.change,
                                    volume: data.volume !== undefined ? data.volume.toLocaleString() : stock.volume,
                                    // Add additional fields for detailed view
                                    priceChange: data.priceChange || stock.priceChange,
                                    bidPrice: data.bidPrice || stock.bidPrice,
                                    askPrice: data.askPrice || stock.askPrice,
                                    bidSize: data.bidSize || stock.bidSize,
                                    askSize: data.askSize || stock.askSize,
                                    tradeTime: data.tradeTime || stock.tradeTime,
                                    lowPrice: data.lowPrice || stock.lowPrice,
                                    openPrice: data.openPrice || stock.openPrice,
                                    highPrice: data.highPrice || stock.highPrice,
                                    previousPrice: data.previousPrice || stock.previousPrice,
                                    averageVolume: data.averageVolume || stock.averageVolume,
                                    stochasticK14d: data.stochasticK14d || stock.stochasticK14d,
                                    weightedAlpha: data.weightedAlpha || stock.weightedAlpha,
                                    priceChange5d: data.priceChange5d || stock.priceChange5d,
                                    percentChange5d: data.percentChange5d || stock.percentChange5d,
                                    lowPrice1y: data.lowPrice1y || stock.lowPrice1y,
                                    highPrice1y: data.highPrice1y || stock.highPrice1y,
                                }
                                : stock
                        );
                        // Save to localStorage
                        saveStocksToLocalStorage(updatedStocks);
                        return updatedStocks;
                    }

                    // Add new stock if it doesn't exist
                    const newStock = {
                        symbol: data.symbol || symbol,
                        companyName: data.symbolName || symbol,
                        price: data.lastPrice !== undefined ? data.lastPrice : 'N/A',
                        change: data.percentChange !== undefined ? (data.percentChange * 100) : 0, // Convert to percentage
                        volume: data.volume !== undefined ? data.volume.toLocaleString() : '0',
                        // Initialize additional fields
                        priceChange: data.priceChange || 'N/A',
                        bidPrice: data.bidPrice || 'N/A',
                        askPrice: data.askPrice || 'N/A',
                        bidSize: data.bidSize || 'N/A',
                        askSize: data.askSize || 'N/A',
                        tradeTime: data.tradeTime || 'N/A',
                        lowPrice: data.lowPrice || 'N/A',
                        openPrice: data.openPrice || 'N/A',
                        highPrice: data.highPrice || 'N/A',
                        previousPrice: data.previousPrice || 'N/A',
                        averageVolume: data.averageVolume || 'N/A',
                        stochasticK14d: data.stochasticK14d || 'N/A',
                        weightedAlpha: data.weightedAlpha || 'N/A',
                        priceChange5d: data.priceChange5d || 'N/A',
                        percentChange5d: data.percentChange5d || 'N/A',
                        lowPrice1y: data.lowPrice1y || 'N/A',
                        highPrice1y: data.highPrice1y || 'N/A',
                    };
                    const updatedStocks = [...prevStocks, newStock];
                    // Save to localStorage
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

    // Function to fetch all stock data with controlled rate limiting
    const fetchAllStockData = useCallback(async () => {
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
                // Wait for 1.5 seconds between requests to respect rate limits
                await new Promise(resolve => setTimeout(resolve, 1500));
            }
        } catch (error) {
            console.error('Error in fetchAllStockData:', error);
        } finally {
            isFetchingRef.current = false;
            setLoading(false); // Only set loading to false after initial load
            setUpdating(false);
        }
    }, [stockSymbols]);

    useEffect(() => {
        // Load data from localStorage on mount
        const savedStocks = loadStocksFromLocalStorage();
        if (savedStocks.length > 0) {
            setStocks(savedStocks);
            setLoading(false);
        }

        // Fetch new data
        fetchAllStockData();

        // Set up interval for periodic updates
        const interval = setInterval(() => {
            fetchAllStockData();
        }, 15 * 60 * 1000); // Refresh every 15 minutes

        return () => clearInterval(interval); // Cleanup on unmount
    }, [fetchAllStockData]);

    // Handler to toggle expanded state
    const handleRowClick = async (symbol) => {
        const isExpanded = expandedRows.includes(symbol);
        if (isExpanded) {
            setExpandedRows(expandedRows.filter(item => item !== symbol));
        } else {
            setExpandedRows([...expandedRows, symbol]);
            // Fetch chart data with a fixed range '1d' if not already fetched
            if (!chartData[symbol]) {
                await fetchChartData(symbol);
            }
        }
    };

    // Function to fetch chart data from Yahoo Finance API with fixed range '1d'
    const fetchChartData = async (symbol) => {
        setLoadingChart(true);
        const fixedRange = '1d'; // Fixed range
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

                // Convert UNIX timestamps to readable format
                const formattedTimestamps = timestamps.map(ts => {
                    const date = new Date(ts * 1000);
                    return `${date.getHours()}:${date.getMinutes().toString().padStart(2, '0')}`;
                });

                // Prepare chart data with enhanced configurations
                const chartConfig = {
                    labels: formattedTimestamps,
                    datasets: [
                        {
                            label: `${symbol} Price`,
                            data: closes,
                            fill: true,
                            backgroundColor: 'rgba(26, 115, 232, 0.2)', // Light blue fill
                            borderColor: '#1A73E8', // Blue line
                            borderWidth: 2,
                            pointRadius: 0, // Removes points for a cleaner line
                            tension: 0.4, // Smooth curves
                            pointHoverRadius: 5, // Points appear on hover
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
                            display: false, // Hide legend for a cleaner look
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
                                label: function(context) {
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
                                display: false, // Hide vertical gridlines
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
                                color: 'rgba(0, 0, 0, 0.05)', // Light horizontal gridlines
                            },
                            ticks: {
                                color: '#333333',
                                font: {
                                    size: 12,
                                },
                                callback: function(value) {
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
        <div className="stock-table-container">
            <h2>Stock Market Prices</h2>
            {loading && (
                <div className="loading-indicator" aria-label="Loading">
                    <div className="spinner"></div>
                </div>
            )}
            {updating && !loading && (
                <div className="updating-indicator" aria-label="Updating">
                    <div className="spinner"></div>
                    <span>Updating Stock Prices...</span>
                </div>
            )}
            <div className="table-responsive">
                <table className="stock-table">
                    <thead>
                        <tr>
                            <th>Stock Code</th>
                            <th>Company Name</th>
                            <th>Current Price</th>
                            <th>Change (%)</th>
                            <th>Volume</th>
                        </tr>
                    </thead>
                    <tbody>
                        {stocks.map((stock) => (
                            <React.Fragment key={stock.symbol}>
                                <tr
                                    className="stock-row"
                                    onClick={() => handleRowClick(stock.symbol)}
                                    tabIndex={0}
                                    onKeyPress={(e) => {
                                        if (e.key === 'Enter' || e.key === ' ') {
                                            handleRowClick(stock.symbol);
                                        }
                                    }}
                                    aria-expanded={expandedRows.includes(stock.symbol)}
                                    role="button"
                                >
                                    <td>
                                        {stock.symbol}
                                        {/* Optional: Chevron Icon */}
                                        <span className={`chevron ${expandedRows.includes(stock.symbol) ? 'rotate' : ''}`}>
                                            ▼
                                        </span>
                                    </td>
                                    <td>{stock.companyName}</td>
                                    <td className="price">${stock.price}</td>
                                    <td className={stock.change >= 0 ? 'change-positive' : 'change-negative'}>
                                        {typeof stock.change === 'number' ? stock.change.toFixed(2) : stock.change}%
                                    </td>
                                    <td>{stock.volume}</td>
                                </tr>
                                {expandedRows.includes(stock.symbol) && (
                                    <tr className="expanded-row">
                                        <td colSpan="5">
                                            <div className="expanded-content">
                                                <h4>Detailed Metrics for {stock.symbol}</h4>
                                                <div className="metrics-grid">
                                                    <div><strong>Symbol Name:</strong> {stock.companyName}</div>
                                                    <div><strong>Last Price:</strong> ${stock.lastPrice || 'N/A'}</div>
                                                    <div><strong>Price Change:</strong> {stock.priceChange || 'N/A'}</div>
                                                    <div><strong>Percent Change:</strong> {stock.percentChange ? (stock.percentChange * 100).toFixed(2) + '%' : 'N/A'}</div>
                                                    <div><strong>Bid Price:</strong> ${stock.bidPrice || 'N/A'}</div>
                                                    <div><strong>Ask Price:</strong> ${stock.askPrice || 'N/A'}</div>
                                                    <div><strong>Bid Size:</strong> {stock.bidSize || 'N/A'}</div>
                                                    <div><strong>Ask Size:</strong> {stock.askSize || 'N/A'}</div>
                                                    <div><strong>Trade Time:</strong> {stock.tradeTime || 'N/A'}</div>
                                                    <div><strong>Low Price:</strong> ${stock.lowPrice || 'N/A'}</div>
                                                    <div><strong>Open Price:</strong> ${stock.openPrice || 'N/A'}</div>
                                                    <div><strong>High Price:</strong> ${stock.highPrice || 'N/A'}</div>
                                                    <div><strong>Previous Price:</strong> ${stock.previousPrice || 'N/A'}</div>
                                                    <div><strong>Average Volume:</strong> {stock.averageVolume || 'N/A'}</div>
                                                    <div><strong>Stochastic K14d:</strong> {stock.stochasticK14d || 'N/A'}</div>
                                                    <div><strong>Weighted Alpha:</strong> {stock.weightedAlpha || 'N/A'}</div>
                                                    <div><strong>Price Change (5d):</strong> {stock.priceChange5d || 'N/A'}</div>
                                                    <div><strong>Percent Change (5d):</strong> {stock.percentChange5d ? (stock.percentChange5d * 100).toFixed(2) + '%' : 'N/A'}</div>
                                                    <div><strong>Low Price (1y):</strong> ${stock.lowPrice1y || 'N/A'}</div>
                                                    <div><strong>High Price (1y):</strong> ${stock.highPrice1y || 'N/A'}</div>
                                                </div>
                                                {/* Chart Section */}
                                                <div className="chart-container">
                                                    <h4>1 Day Price Chart</h4>
                                                    {loadingChart && (
                                                        <div className="loading-indicator" aria-label="Loading Chart">
                                                            <div className="spinner"></div>
                                                        </div>
                                                    )}
                                                    {chartData[stock.symbol] ? (
                                                        <Line data={chartData[stock.symbol].data} options={chartData[stock.symbol].options} />
                                                    ) : (
                                                        !loadingChart && <p>No chart data available.</p>
                                                    )}
                                                </div>
                                            </div>
                                        </td>
                                    </tr>
                                )}
                            </React.Fragment>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );

};

export default StockTable;
