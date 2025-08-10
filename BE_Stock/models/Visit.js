// models/Visit.js
const mongoose = require('mongoose');

const VisitSchema = new mongoose.Schema({
    count: {
        type: Number,
        default: 0,
    },
}, { timestamps: true });

module.exports = mongoose.model('Visit', VisitSchema);
