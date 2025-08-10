import React from 'react';
import '../components/StockTable.css';
import { Line } from 'react-chartjs-2';

const StockTable = ({
    stocks,
    loading,
    updating,
    expandedRows,
    handleRowClick,
    chartData,
    loadingChart,
    isAdmin,
}) => {
    return (
        <div className="stock-table-container">
            <h2>Stock Market Prices {isAdmin && "(Admin)"}</h2>
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
                            {isAdmin && <th>Actions</th>} {/* Add Actions column for Admin */}
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
                                    {isAdmin && (
                                        <td>
                                            <button onClick={(e) => { e.stopPropagation(); /* handle edit */ }}>Edit</button>
                                            <button onClick={(e) => { e.stopPropagation(); /* handle delete */ }}>Delete</button>
                                        </td>
                                    )}
                                </tr>
                                {expandedRows.includes(stock.symbol) && (
                                    <tr className="expanded-row">
                                        <td colSpan={isAdmin ? "6" : "5"}>
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
                                                {/* Chart section */}
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