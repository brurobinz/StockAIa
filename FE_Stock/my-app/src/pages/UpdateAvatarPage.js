import React, { useState } from 'react';
import axios from 'axios';
import { useParams, useNavigate } from 'react-router-dom';

const UpdateAvatarPage = () => {
    const { id } = useParams();
    const navigate = useNavigate();
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        setSelectedFile(file);

        if (file) {
            const fileUrl = URL.createObjectURL(file);
            setPreviewUrl(fileUrl);
        }
    };

    const handleUpload = async () => {
        if (!selectedFile) {
            alert("Please select an image first.");
            return;
        }
    
        const formData = new FormData();
        formData.append('avatar', selectedFile);
    
        try {
            const response = await axios.post(`http://localhost:4000/api/users/${id}/upload-avatar`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                    'Authorization': `Bearer ${localStorage.getItem('token')}`,
                },
            });
            alert("Avatar updated successfully!");
    
            // Save the avatar's URL to localStorage so that the Header can use it
            localStorage.setItem('userAvatar', response.data.avatarUrl);
            
            // Navigate users to the Account Page after a successful upload
            navigate(`/account/${id}`);
        } catch (error) {
            console.error("Error uploading avatar:", error);
            alert("Failed to upload avatar.");
        }
    };    

    return (
        <div style={{ textAlign: 'center', padding: '20px', backgroundColor: '#f8f9fa', borderRadius: '8px', maxWidth: '400px', margin: '0 auto' }}>
            <h2>Update Avatar</h2>
            <div>
                <label>Select Avatar:</label>
                <input type="file" accept="image/*" onChange={handleFileChange} />
            </div>
            {previewUrl && (
                <div style={{ marginTop: '15px' }}>
                    <p>Preview:</p>
                    <img src={previewUrl} alt="Avatar Preview" style={{ width: '100px', height: '100px', borderRadius: '50%' }} />
                </div>
            )}
            <button onClick={handleUpload} style={{ marginTop: '20px', padding: '10px 20px', backgroundColor: '#007bff', color: '#fff', border: 'none', borderRadius: '5px', cursor: 'pointer' }}>
                Upload Avatar
            </button>
        </div>
    );
};

export default UpdateAvatarPage;