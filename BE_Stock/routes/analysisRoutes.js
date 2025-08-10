const express = require('express');
const router = express.Router();
const analysisController = require('../controllers/analysisController');

router.post('/analyses', analysisController.createAnalysis);
router.get('/analyses', analysisController.getAnalyses);
router.get('/analyses/:id', analysisController.getAnalysisById);

module.exports = router;
