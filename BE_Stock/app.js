require('dotenv').config(); // Load environment variables

const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const multer = require('multer');
const path = require('path');

// Import routes
const userRoutes = require('./routes/userRoutes');
const stockRoutes = require('./routes/stockRoutes');
const stockPriceRoutes = require('./routes/stockPriceRoutes');
const newsRoutes = require('./routes/newsRoutes');
const analysisRoutes = require('./routes/analysisRoutes');
const chatRoutes = require('./routes/chatRoutes'); // Import chat routes
const messageRoutes = require('./routes/messageRouters');

// Import middleware
const { auth, admin } = require('./middleware/auth'); // Import Authentication Middleware

// Import models
const User = require('./models/User'); // Import User model

const app = express();

// Middleware for JSON and CORS analysis
app.use(express.json());
app.use(cors());

// Configure multer for file uploads with file type and size checks
const upload = multer({ 
    dest: 'uploads/',
    limits: { fileSize: 5 * 1024 * 1024 }, // File size limit 5MB
    fileFilter: (req, file, cb) => {
        const filetypes = /jpeg|jpg|png/;
        const mimetype = filetypes.test(file.mimetype);
        const extname = filetypes.test(path.extname(file.originalname).toLowerCase());

        if (mimetype && extname) {
            return cb(null, true);
        }
        cb(new Error('Only JPEG, JPG, and PNG files are allowed.'));
    }
});

// Endpoint to handle avatar uploads (protected with auth middleware)
app.put('/api/users/:id/avatar', auth, upload.single('avatar'), async (req, res) => { // Apply auth middleware
    const userId = req.params.id;

    // Make sure that the authenticated user is the one who is updating their avatar
    if (req.user._id.toString() !== userId) {
        return res.status(403).json({ message: 'You can only update your own avatar.' });
    }

    if (req.file) {
        console.log('File uploaded:', req.file); // Debug file

        try {
            // Update the user's avatar field in the database
            const updatedUser = await User.findByIdAndUpdate(
                userId,
                { avatar: `/uploads/${req.file.filename}` },
                { new: true, runValidators: true }
            ).select('-password'); // Remove the password field from the response

            res.status(200).json({ message: 'Avatar uploaded successfully', user: updatedUser });
        } catch (err) {
            console.error('Error updating avatar:', err);
            res.status(500).json({ message: 'Server error when updating avatar.' });
        }
    } else {
        res.status(400).json({ message: 'No files have been uploaded.' });
    }
});

// Mount routes
app.use('/api/users', userRoutes);
app.use('/api/stocks', stockRoutes);
app.use('/api/stock-prices', stockPriceRoutes);
app.use('/api/news', newsRoutes);
app.use('/api/analysis', analysisRoutes);
app.use('/api/chat', chatRoutes); // Mount chat routes
app.use('/api/message', messageRoutes);

// Serve uploaded files statically
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// Connect to MongoDB using a URI from an .env file
mongoose.connect(process.env.MONGODB_URI, { useNewUrlParser: true, useUnifiedTopology: true })
    .then(() => {
        console.log('MongoDB connected');
        const PORT = process.env.PORT || 4000;
        app.listen(PORT, () => {
            console.log(`Server running on port ${PORT}`);
        });
    })
    .catch(err => {
        console.error('MongoDB connection error:', err);
        process.exit(1);
    });

// Middleware handles common errors
app.use((err, req, res, next) => {
    console.error('Unhandled Error:', err.stack);
    res.status(500).send('Something broke!');
});