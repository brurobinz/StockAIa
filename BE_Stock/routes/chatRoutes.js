// routes/chatRoutes.js
const express = require('express');
const router = express.Router();
const ChatMessage = require('../models/ChatMessage');
const ChatSession = require('../models/ChatSession');
const { auth } = require('../middleware/auth'); // Import middleware xác thực

// @route   GET /api/chat/sessions
// @desc    Lấy danh sách các phiên trò chuyện của người dùng
// @access  Private
router.get('/sessions', auth, async (req, res) => {
    try {
        const sessions = await ChatSession.find({ user: req.user._id }).sort({ createdAt: -1 });
        res.json(sessions);
    } catch (error) {
        console.error('Error fetching chat sessions:', error);
        res.status(500).json({ message: 'Server error' });
    }
});

// @route   POST /api/chat/sessions
// @desc    Tạo một phiên trò chuyện mới
// @access  Private
router.post('/sessions', auth, async (req, res) => {
    const { title } = req.body;
    try {
        const newSession = new ChatSession({
            user: req.user._id,
            title: title || 'New Chat Session',
        });
        await newSession.save();
        res.status(201).json(newSession);
    } catch (error) {
        console.error('Error creating chat session:', error);
        res.status(500).json({ message: 'Server error' });
    }
});

// @route   GET /api/chat/sessions/:sessionId/messages
// @desc    Lấy tin nhắn trong một phiên trò chuyện cụ thể
// @access  Private
router.get('/sessions/:sessionId/messages', auth, async (req, res) => {
    const { sessionId } = req.params;
    try {
        // Kiểm tra phiên trò chuyện có thuộc về người dùng hay không
        const session = await ChatSession.findOne({ _id: sessionId, user: req.user._id });
        if (!session) {
            return res.status(404).json({ message: 'Chat session not found' });
        }

        const messages = await ChatMessage.find({ chatSession: sessionId }).sort({ timestamp: 1 });
        res.json(messages);
    } catch (error) {
        console.error('Error fetching chat messages:', error);
        res.status(500).json({ message: 'Server error' });
    }
});

// @route   POST /api/chat/sessions/:sessionId/messages
// @desc    Gửi một tin nhắn trong phiên trò chuyện cụ thể
// @access  Private
router.post('/sessions/:sessionId/messages', auth, async (req, res) => {
    const { sessionId } = req.params;
    const { text, sender } = req.body;

    if (!text || !sender) {
        return res.status(400).json({ message: 'Text and sender are required' });
    }

    try {
        // Kiểm tra phiên trò chuyện có thuộc về người dùng hay không
        const session = await ChatSession.findOne({ _id: sessionId, user: req.user._id });
        if (!session) {
            return res.status(404).json({ message: 'Chat session not found' });
        }

        const newMessage = new ChatMessage({
            chatSession: sessionId,
            text,
            sender,
        });
        await newMessage.save();
        res.status(201).json(newMessage);
    } catch (error) {
        console.error('Error saving chat message:', error);
        res.status(500).json({ message: 'Server error' });
    }
});

module.exports = router;
