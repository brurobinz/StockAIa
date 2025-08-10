const StockPrice = require('../models/StockPrice');

exports.createStockPrice = async (req, res) => {
  const stockPrice = new StockPrice(req.body);
  await stockPrice.save();
  res.status(201).send(stockPrice);
};

exports.getStockPrices = async (req, res) => {
  const stockPrices = await StockPrice.find().populate('stock_id');
  res.status(200).send(stockPrices);
};

exports.getStockPriceById = async (req, res) => {
  const stockPrice = await StockPrice.findById(req.params.id).populate('stock_id');
  res.status(200).send(stockPrice);
};