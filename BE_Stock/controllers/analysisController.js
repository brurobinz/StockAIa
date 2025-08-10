const Analysis = require('../models/Analysis');

exports.createAnalysis = async (req, res) => {
  const analysis = new Analysis(req.body);
  await analysis.save();
  res.status(201).send(analysis);
};

exports.getAnalyses = async (req, res) => {
  const analyses = await Analysis.find().populate('stock_id');
  res.status(200).send(analyses);
};

exports.getAnalysisById = async (req, res) => {
  const analysis = await Analysis.findById(req.params.id).populate('stock_id');
  res.status(200).send(analysis);
};
