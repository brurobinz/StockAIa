// models/ChatMessage.js
const mongoose = require('mongoose');

const ChatMessageSchema = new mongoose.Schema({
    chatSession: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'ChatSession',
        required: true,
        index: true,
    },
    text: {
        type: String,
        required: true,
    },
    sender: {
        type: String,
        enum: ['user', 'bot'],
        required: true,
    },
    timestamp: {
        type: Date,
        default: Date.now,
        index: true,
    },
});

// Tạo chỉ mục cho hiệu suất tốt hơn khi truy vấn theo chatSession và timestamp
ChatMessageSchema.index({ chatSession: 1, timestamp: 1 });

module.exports = mongoose.model('ChatMessage', ChatMessageSchema);
