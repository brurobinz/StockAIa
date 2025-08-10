const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const analysisSchema = new Schema({
  stock_id: { type: Schema.Types.ObjectId, ref: 'Stock' },
  analysis_type: String,
  description: String,
  created_by: String,
});

module.exports = mongoose.model('Analysis', analysisSchema);