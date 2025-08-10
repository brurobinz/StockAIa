const yahooFinance = require('yahoo-finance2').default;

// Function to get stock price by symbol
const getStockPrice = async (req, res) => {
    try {
        const { symbol } = req.params; // Get stock symbol from URL
        const data = await yahooFinance.quote(symbol); // Call Yahoo Finance API
        res.json(data); // Return data to frontend
    } catch (error) {
        console.error("Error fetching stock price:", error);
        res.status(500).json({ error: "Failed to fetch stock price data" });
    }
};

// Define dummy functions for `createStock`, `getStocks`, and `getStockById`
const createStock = (req, res) => {
    // Replace with your logic to create a stock
    res.json({ message: "Stock created successfully" });
};

const getStocks = (req, res) => {
    // Replace with your logic to retrieve all stocks
    res.json({ stocks: [] });
};

const getStockById = (req, res) => {
    // Replace with your logic to retrieve a stock by ID
    const { id } = req.params;
    res.json({ stockId: id });
};

module.exports = { getStockPrice, createStock, getStocks, getStockById };
