const User = require('../models/User');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const mongoose = require('mongoose');
const cloudinary = require('cloudinary').v2;
const fs = require('fs');

// Configure Cloudinary
cloudinary.config({
    cloud_name: 'dq2ucdguz',
    api_key: '622122876857883',
    api_secret: '7qpEjz_9Qj-j03rCEPCi961un7U',
});

// Helper function to upload avatar to Cloudinary
async function uploadAvatar(filePath) {
    try {
        const result = await cloudinary.uploader.upload(filePath, { folder: "avatars" });
        return result.secure_url; // URL of the uploaded avatar
    } catch (error) {
        console.error("Error uploading avatar:", error);
        throw new Error('Avatar upload failed');
    }
}

// Register a new user
exports.register = async (req, res) => {
    const { username, email, password, dateOfBirth } = req.body;

    try {
        // Check if user already exists
        let user = await User.findOne({ email });
        if (user) {
            return res.status(400).json({ message: 'User already exists' });
        }

        user = new User({
            username,
            email,
            password,
            dateOfBirth,
        });

        // Hash the password
        const salt = await bcrypt.genSalt(10);
        user.password = await bcrypt.hash(password, salt);

        await user.save();

        // Create JWT Payload
        const payload = {
            user: {
                id: user.id,
            },
        };

        // Sign JWT
        jwt.sign(
            payload,
            process.env.JWT_SECRET,
            { expiresIn: '1h' },
            (err, token) => {
                if (err) throw err;
                res.status(201).json({ token, userId: user.id }); // Return token and userId
            }
        );
    } catch (error) {
        console.error('Error in register:', error.message);
        res.status(500).send('Server Error');
    }
};

// User login
exports.login = async (req, res) => {
    const { email, password } = req.body;

    try {
        // Check if user exists
        const user = await User.findOne({ email });
        if (!user) {
            console.error('User not found with email:', email);
            return res.status(400).json({ message: 'User or Password not found!' });
        }

        // Check if user is inactive before checking the password
        if (user.status === 'Inactive') {
            console.error('Inactive user trying to log in:', email);
            return res.status(403).json({ message: 'Your account is inactive. Please contact support.' });
        }

        // Proper bcrypt comparison
        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch) {
            console.error('Password does not match for email:', email);
            return res.status(400).json({ message: 'User or Password not found!' });
        }

        // Create JWT Payload
        const payload = {
            user: {
                id: user.id,
                role: user.role, // Include role in payload
            },
        };

        // Sign JWT
        jwt.sign(
            payload,
            process.env.JWT_SECRET,
            { expiresIn: '1h' },
            (err, token) => {
                if (err) throw err;
                res.json({ token, userId: user.id, role: user.role }); // Include role in response
            }
        );
    } catch (error) {
        console.error('Error in login:', error.message);
        res.status(500).send('Server Error');
    }
};

// Forgot password
exports.forgotPassword = async (req, res) => {
    // Implement forgot password logic here (e.g., send reset email)
    res.status(200).json({ msg: 'Password reset link sent' });
};

// Reset password
exports.resetPassword = async (req, res) => {
    // Implement reset password logic here
    res.status(200).json({ msg: 'Password reset successfully' });
};

// Get current user's profile
exports.getProfile = async (req, res) => {
    try {
        const userId = req.user.id;

        if (!mongoose.Types.ObjectId.isValid(userId)) {
            return res.status(400).json({ msg: 'Invalid user ID' });
        }

        const user = await User.findById(userId).select('-password');
        if (!user) {
            return res.status(404).json({ msg: 'User not found' });
        }

        res.status(200).json({
            id: user.id,
            username: user.username,
            email: user.email,
            role: user.role, // Include role in response
            firstName: user.firstName,
            lastName: user.lastName,
            preferredName: user.preferredName,
            gender: user.gender,
            phoneNumber: user.phoneNumber,
            country: user.country,
            dateOfBirth: user.dateOfBirth,
            avatar: user.avatar,
            status: user.status
        });
    } catch (error) {
        console.error('Error in getProfile:', error);
        res.status(500).send('Server Error');
    }
};

// Update current user's profile
exports.updateProfile = async (req, res) => {
    const {
        firstName,
        lastName,
        preferredName,
        gender,
        phoneNumber,
        country,
        username,
        email,
        dateOfBirth
    } = req.body;

    try {
        const userId = req.params.id;

        // Kiểm tra xem người dùng có quyền cập nhật thông tin này không
        if (req.user.id !== userId && req.user.role !== 'admin') {
            return res.status(403).json({ msg: 'Access denied: You can only update your own profile' });
        }

        if (!mongoose.Types.ObjectId.isValid(userId)) {
            return res.status(400).json({ msg: 'Invalid user ID' });
        }

        let user = await User.findById(userId);
        if (!user) {
            return res.status(404).json({ msg: 'User not found' });
        }

        // Kiểm tra trùng lặp email và username nếu cần
        if (email && email !== user.email) {
            const existingEmail = await User.findOne({ email });
            if (existingEmail) {
                return res.status(400).json({ msg: 'Email đã được sử dụng' });
            }
        }

        if (username && username !== user.username) {
            const existingUsername = await User.findOne({ username });
            if (existingUsername) {
                return res.status(400).json({ msg: 'Username đã được sử dụng' });
            }
        }

        // Update fields if provided
        if (firstName !== undefined) user.firstName = firstName;
        if (lastName !== undefined) user.lastName = lastName;
        if (preferredName !== undefined) user.preferredName = preferredName;
        if (gender !== undefined) user.gender = gender;
        if (phoneNumber !== undefined) user.phoneNumber = phoneNumber;
        if (country !== undefined) user.country = country;
        if (username !== undefined) user.username = username;
        if (email !== undefined) user.email = email;
        if (dateOfBirth !== undefined) {
            if (isNaN(Date.parse(dateOfBirth))) {
                return res.status(400).json({ msg: 'Invalid date of birth' });
            }
            user.dateOfBirth = new Date(dateOfBirth);
        }

        // Handle avatar upload if file is present
        if (req.file) {
            const avatarUrl = await uploadAvatar(req.file.path);
            user.avatar = avatarUrl;
            fs.unlink(req.file.path, (err) => {
                if (err) console.error("Error deleting temp file:", err);
            });
        }

        await user.save();

        res.status(200).json({
            msg: 'Profile updated successfully',
            user: {
                id: user.id,
                username: user.username,
                email: user.email,
                role: user.role,
                firstName: user.firstName,
                lastName: user.lastName,
                preferredName: user.preferredName,
                gender: user.gender,
                phoneNumber: user.phoneNumber,
                country: user.country,
                dateOfBirth: user.dateOfBirth,
                avatar: user.avatar,
                status: user.status
            }
        });
    } catch (error) {
        console.error('Error in updateProfile:', error);
        res.status(500).json({ msg: 'Server Error', error: error.message });
    }
};

// Get user by ID (public)
exports.getUserById = async (req, res) => {
    try {
        const userId = req.params.id; // Sử dụng ':id' như đã định nghĩa trong route
        console.log('Received userId:', userId); // Thêm log này để kiểm tra

        if (!mongoose.Types.ObjectId.isValid(userId)) {
            return res.status(400).json({ msg: 'Invalid user ID' });
        }

        const user = await User.findById(userId).select('-password');
        if (!user) {
            return res.status(404).json({ msg: 'User not found' });
        }
        res.status(200).json({
            _id: user._id, // Thêm _id vào phản hồi
            id: user.id,
            username: user.username,
            email: user.email,
            role: user.role, // Include role in response
            firstName: user.firstName,
            lastName: user.lastName,
            preferredName: user.preferredName,
            gender: user.gender,
            phoneNumber: user.phoneNumber,
            country: user.country,
            dateOfBirth: user.dateOfBirth,
            avatar: user.avatar,
            status: user.status
        });
    } catch (error) {
        console.error('Error in getUserById:', error);
        res.status(500).send('Server Error');
    }
};

// Upload user avatar
exports.uploadUserAvatar = async (req, res) => {
    const userId = req.params.id; // Use req.params.id instead of req.user.id

    try {
        if (!req.file) {
            return res.status(400).json({ msg: 'No file uploaded' });
        }

        // Upload avatar to Cloudinary
        const avatarUrl = await uploadAvatar(req.file.path);

        // Find user and update avatar URL
        const user = await User.findById(userId);
        if (!user) {
            return res.status(404).json({ msg: 'User not found' });
        }

        user.avatar = avatarUrl;
        await user.save();

        // Delete temporary file after successful upload
        fs.unlink(req.file.path, (err) => {
            if (err) console.error("Error deleting temp file:", err);
        });

        res.status(200).json({ msg: 'Avatar updated successfully', avatarUrl });
    } catch (error) {
        console.error('Error uploading avatar:', error);
        res.status(500).json({ msg: 'Server Error' });
    }
};

// Get all users (Admin only)
exports.getAllUsers = async (req, res) => {
    try {
        const users = await User.find().select('-password').sort({ createdAt: -1 });
        res.status(200).json(users.map(user => ({
            _id: user._id, // Thêm _id vào phản hồi
            id: user.id,
            username: user.username,
            email: user.email,
            role: user.role, // Include role in response
            firstName: user.firstName,
            lastName: user.lastName,
            preferredName: user.preferredName,
            gender: user.gender,
            phoneNumber: user.phoneNumber,
            country: user.country,
            dateOfBirth: user.dateOfBirth,
            avatar: user.avatar,
            status: user.status,
            walletBalance: user.walletBalance || 0, // 👈 Thêm dòng này
            walletHistory: user.walletHistory || [],
        })));
    } catch (err) {
        console.error('Error fetching users:', err);
        res.status(500).json({ msg: 'Server Error' });
    }
};

// Update any user's information (Admin only)
exports.updateUser = async (req, res) => {
    const userId = req.params.id;
    const {
        username,
        email,
        role,
        firstName,
        lastName,
        preferredName,
        gender,
        phoneNumber,
        country,
        dateOfBirth,
        avatar
    } = req.body;

    try {
        if (!mongoose.Types.ObjectId.isValid(userId)) {
            return res.status(400).json({ msg: 'Invalid user ID' });
        }

        let user = await User.findById(userId);
        if (!user) {
            return res.status(404).json({ msg: 'User not found' });
        }

        // Update fields if provided
        if (username !== undefined) user.username = username;
        if (email !== undefined) user.email = email;
        if (role !== undefined) user.role = role;
        if (firstName !== undefined) user.firstName = firstName;
        if (lastName !== undefined) user.lastName = lastName;
        if (preferredName !== undefined) user.preferredName = preferredName;
        if (gender !== undefined) user.gender = gender;
        if (phoneNumber !== undefined) user.phoneNumber = phoneNumber;
        if (country !== undefined) user.country = country;
        if (dateOfBirth !== undefined) {
            if (isNaN(Date.parse(dateOfBirth))) {
                return res.status(400).json({ msg: 'Invalid date of birth' });
            }
            user.dateOfBirth = new Date(dateOfBirth);
        }
        if (avatar !== undefined) user.avatar = avatar;

        await user.save();

        res.status(200).json({ msg: 'User updated successfully', user });
    } catch (error) {
        console.error('Error in updateUser:', error);
        res.status(500).json({ msg: 'Server Error', error: error.message });
    }
};

exports.updateStatus = async (req, res) => {
    try {
        const userId = req.params.id;
        const { status } = req.body;

        const user = await User.findById(userId);
        if (!user) {
            return res.status(404).json({ msg: 'User not found' });
        }

        user.status = status;
        await user.save();

        res.status(200).json({ msg: 'User status updated successfully', status: user.status });
    } catch (error) {
        console.error('Error in updateStatus:', error);
        res.status(500).json({ msg: 'Server Error', error: error.message });
    }
};

exports.countUsers = async (req, res) => {
    try {
        const count = await User.countDocuments();
        res.status(200).json({ count });
    } catch (error) {
        console.error('Error fetching user count:', error);
        res.status(500).json({ msg: 'Server Error' });
    }
};
exports.updatePassword = async (req, res) => {
    try {
        const userId = req.params.id;
        const { oldPassword, newPassword } = req.body;

        if (!newPassword || !oldPassword) {
            return res.status(400).json({ message: 'Both old and new password are required' });
        }

        const user = await User.findById(userId);
        if (!user) {
            return res.status(404).json({ message: 'User not found' });
        }

        const isMatch = await bcrypt.compare(oldPassword, user.password);
        if (!isMatch) {
            return res.status(401).json({ message: 'Old password is incorrect' });
        }

        const hashedPassword = await bcrypt.hash(newPassword, 10);
        user.password = hashedPassword;
        await user.save();

        res.json({ message: 'Password updated successfully' });
    } catch (error) {
        console.error('Error updating password:', error);
        res.status(500).json({ message: 'Internal server error' });
    }
};
exports.depositToWallet = async (req, res) => {
    try {
        const { id } = req.params;
        const { amount } = req.body;

        if (!amount || amount <= 0) {
            return res.status(400).json({ message: 'Invalid amount' });
        }

        const user = await User.findById(id);
        if (!user) return res.status(404).json({ message: 'User not found' });

        user.walletBalance += amount;
        user.walletHistory.push({
            type: 'Deposit',
            amount,
        });

        await user.save();

        res.status(200).json({
            message: 'Deposit successful',
            walletBalance: user.walletBalance,
            walletHistory: user.walletHistory,
        });
    } catch (err) {
        res.status(500).json({ message: 'Server error' });
    }
};
