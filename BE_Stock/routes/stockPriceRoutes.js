// routes/stockPriceRoutes.js
const express = require('express');
const { getStockPrice } = require('../controllers/stockController');

const router = express.Router();

// Route để lấy giá cổ phiếu theo mã
router.get('/:symbol', getStockPrice);

module.exports = router;
