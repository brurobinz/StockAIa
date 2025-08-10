import React, { useState } from "react";
import Navbar from "../components/common/Navbar";
import Footer from "../components/common/Footer";
import Chart from "../components/Chart";
import "./AnalysisPage.css"; // Import the CSS file for styling

const Analysis = () => {
    const [ticker, setTicker] = useState("TSLA");
    const [method, setMethod] = useState("ARIMA");
    const [period, setPeriod] = useState(1);
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handlePredict = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(
                `http://127.0.0.1:8000/predict?ticker=${ticker}&method=${method}&period=${period}`
            );
            const result = await response.json();
    
            if (!result || !result.dates || !result.prices) {
                throw new Error("Invalid data received from API");
            }
    
            // Set data for visualization
            setData({
                dates: result.dates,
                prices: result.prices,
                historicalLength: result.historicalLength,
            });
        } catch (err) {
            setError("Error fetching forecasted data. Please try again.");
        } finally {
            setLoading(false);
        }
    };
    
    
    return (
        <div className="analysis-container">
            <Navbar />
            <div className="analysis-content">
                {/* Controls Section */}
                <div className="controls-section">
                    <div className="controls-card">
                        <h2 className="controls-title">Stock Price Prediction</h2>
                        <div className="controls-form">
                            <div className="form-group">
                                <label className="form-label">Select Stock</label>
                                <select
                                    value={ticker}
                                    onChange={(e) => setTicker(e.target.value)}
                                    className="form-select"
                                >
                                    <option value="TSLA">Tesla (TSLA)</option>
                                    <option value="MSFT">Microsoft (MSFT)</option>
                                    <option value="GOOGL">Google (GOOGL)</option>
                                    <option value="AMZN">Amazon (AMZN)</option>
                                    <option value="AAPL">Apple (AAPL)</option>
                                </select>
                            </div>
                            <div className="form-group">
                                <label className="form-label">Prediction Method</label>
                                <select
                                    value={method}
                                    onChange={(e) => setMethod(e.target.value)}
                                    className="form-select"
                                >
                                    <option value="ARIMA">ARIMA</option>
                                    <option value="Prophet">Prophet</option>
                                    <option value="LSTM">LSTM</option>
                                    <option value="GRU">GRU</option>
                                    <option value="XGBoost">XGBoost</option>
                                    <option value="Hybrid">Hybrid GRU + LSTM</option>
                                </select>
                            </div>
                            <div className="form-group">
                                <label className="form-label">Forecast Period</label>
                                <select
                                    value={period}
                                    onChange={(e) => setPeriod(Number(e.target.value))}
                                    className="form-select"
                                >
                                    <option value={1}>1 Day</option>
                                    <option value={7}>1 Week</option>
                                    <option value={30}>1 Month</option>
                                </select>
                            </div>
                            <button
                                onClick={handlePredict}
                                disabled={loading}
                                className={`predict-button ${loading ? "disabled-button" : ""}`}
                            >
                                {loading ? "Loading..." : "Predict"}
                            </button>
                            {error && <p className="error-message">{error}</p>}
                        </div>
                    </div>
                </div>

                {/* Chart and Table Section */}
                <div className="visualization-section">
                    <div className="chart-card">
                        <h3 className="chart-title">Stock Price Chart for {ticker}</h3>
                        {data ? (
                            <Chart data={data} />
                        ) : (
                            <p className="chart-placeholder">
                                {loading
                                    ? "Fetching chart data..."
                                    : "Select options and click Predict to see the chart."}
                            </p>
                        )}
                    </div>
                    <div className="table-card">
                        <h3 className="table-title">Forecasted Data</h3>
                        {data ? (
                            <table className="forecast-table">
                                <thead>
                                    <tr>
                                        <th>Date</th>
                                        <th>Forecasted Price ($)</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {data.dates.slice(data.historicalLength).map((date, index) => (
                                        <tr key={index}>
                                            <td>{date}</td>
                                            <td>
                                                {data.prices[data.historicalLength + index] !== null
                                                    ? data.prices[data.historicalLength + index].toFixed(2)
                                                    : "-"}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        ) : (
                            <p className="table-placeholder">
                                {loading
                                    ? "Fetching table data..."
                                    : "Select options and click Predict to see the forecast."}
                            </p>
                        )}
                    </div>
                </div>
            </div>
            <Footer />
        </div>
    );
};

export default Analysis;
