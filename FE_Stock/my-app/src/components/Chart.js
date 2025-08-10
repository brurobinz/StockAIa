import React, { useRef } from "react";
import { Line } from "react-chartjs-2";
import {
    Chart as ChartJS,
    LineElement,
    CategoryScale,
    LinearScale,
    PointElement,
    Tooltip,
    Legend,
    Title,
    Filler,
} from "chart.js";
import zoomPlugin from "chartjs-plugin-zoom";
import { FiZoomIn, FiZoomOut, FiRefreshCw } from "react-icons/fi";
import "./Chart.css";

// Register Chart.js components and zoom plugin
ChartJS.register(
    LineElement,
    CategoryScale,
    LinearScale,
    PointElement,
    Tooltip,
    Legend,
    Title,
    Filler,
    zoomPlugin
);

const Chart = ({ data }) => {
    const chartRef = useRef(null);


    // Prepare the chart data
    const chartData = {
        labels: data?.dates || [],
        datasets: [
            {
                label: "Historical Data (Closing Price)",
                data: data?.prices?.slice(0, data.historicalLength) || [],
                borderColor: "green",
                backgroundColor: "rgba(0, 255, 0, 0.1)",
                borderWidth: 2,
                fill: false,
                pointRadius: 0,
            },
            {
                label: "Forecasted Data (Closing Price)",
                data: [
                    ...(data?.prices
                        ? new Array(data.historicalLength).fill(null)
                        : []),
                    ...(data?.prices?.slice(data.historicalLength) || []),
                ],
                borderColor: "blue",
                backgroundColor: "rgba(0, 0, 255, 0.1)",
                borderWidth: 2,
                borderDash: [5, 5],
                fill: false,
                pointRadius: 4,
                pointBackgroundColor: "red",
            },
        ],
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: {
                title: {
                    display: true,
                    text: "Date",
                },
                ticks: {
                    autoSkip: true,
                    maxTicksLimit: 10,
                },
            },
            y: {
                title: {
                    display: true,
                    text: "Price ($)",
                },
                beginAtZero: false,
            },
        },
        plugins: {
            legend: {
                display: true,
                position: "top",
                labels: {
                    font: {
                        size: 14,
                    },
                },
            },
            title: {
                display: true,
                text: "Stock Price Chart",
                font: {
                    size: 20,
                    weight: "bold",
                },
                padding: {
                    top: 10,
                    bottom: 20,
                },
            },
            zoom: {
                pan: {
                    enabled: true,
                    mode: "x", // Enable panning only on the x-axis
                },
                zoom: {
                    enabled: true,
                    mode: "x", // Enable zooming only on the x-axis
                    drag: true, // Enable drag-to-zoom functionality
                },
            },
        },
    };

    // Zoom controls
    const zoomIn = () => {
        if (chartRef.current) {
            chartRef.current.zoom(1.2); // Zoom in by 20%
        }
    };

    const zoomOut = () => {
        if (chartRef.current) {
            chartRef.current.zoom(0.8); // Zoom out by 20%
        }
    };

    const resetZoom = () => {
        if (chartRef.current) {
            chartRef.current.resetZoom(); // Reset zoom using the plugin
        }
    };

    return (
        <div style={{ position: "relative", width: "100%", height: "600px", marginTop: "20px" }}>
            <Line ref={chartRef} data={chartData} options={options} />
            <div
                style={{
                    position: "absolute",
                    top: "10px",
                    right: "10px",
                    display: "flex",
                    flexDirection: "column",
                    gap: "10px",
                    zIndex: 1,
                }}
            >
                {/* Zoom In Icon */}
                <div onClick={zoomIn} className="chart-icon zoom-in">
                    <FiZoomIn size={20} />
                </div>

                {/* Zoom Out Icon */}
                <div onClick={zoomOut} className="chart-icon zoom-out">
                    <FiZoomOut size={20} />
                </div>

                {/* Reset Zoom Icon */}
                <div onClick={resetZoom} className="chart-icon reset-zoom">
                    <FiRefreshCw size={20} />
                </div>
            </div>
        </div>
    );
};

export default Chart;
