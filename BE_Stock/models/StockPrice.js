const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const stockPriceSchema = new Schema({
  stock_id: { type: Schema.Types.ObjectId, ref: 'Stock' },
  date: Date,
  open_price: Number,
  high_price: Number,
  low_price: Number,
  close_price: Number,
  adj_close_price: Number,
  volume: Number,
});

module.exports = mongoose.model('StockPrice', stockPriceSchema);