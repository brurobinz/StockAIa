const express = require('express');
const router = express.Router();
const Message = require('../models/Message');

// Gửi tin nhắn
router.post('/', async (req, res) => {
    try {
        const { sender, userId, content } = req.body;
        const message = new Message({ sender, userId, content });
        await message.save();
        res.status(201).json({ message: 'Message sent', data: message });
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'Failed to send message' });
    }
});

// Lấy tin nhắn theo user
router.get('/:userId', async (req, res) => {
    try {
        const messages = await Message.find({ userId: req.params.userId });
        res.json(messages);
    } catch (err) {
        res.status(500).json({ error: 'Failed to fetch messages' });
    }
});

// ✅ Lấy tất cả tin nhắn (dành cho Admin)
router.get('/', async (req, res) => {
    try {
        const messages = await Message.find();
        res.json(messages);
    } catch (err) {
        res.status(500).json({ error: 'Failed to fetch all messages' });
    }
});

module.exports = router;
