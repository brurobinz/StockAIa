const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const stockSchema = new Schema({
  symbol: String,
  company_name: String,
  sector: String,
  market: String,
  currency: String,
});

module.exports = mongoose.model('Stock', stockSchema);