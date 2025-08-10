const express = require('express');
const router = express.Router();
const stockController = require('../controllers/stockController');

router.post('/stocks', stockController.createStock);
router.get('/stocks', stockController.getStocks);
router.get('/stocks/:id', stockController.getStockById);

module.exports = router;
