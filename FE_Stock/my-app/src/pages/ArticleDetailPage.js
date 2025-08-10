import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import Navbar from '../components/common/Navbar';
import Footer from '../components/common/Footer';
import { FaArrowLeft } from 'react-icons/fa';
import '../pages/ArticleDetailPage.css';

const ArticleDetailPage = () => {
    const location = useLocation();
    const navigate = useNavigate();
    const article = location.state?.article;

    const handleBack = () => {
        navigate(-1);
    };

    if (!article) {
        return (
            <div>
                <Navbar />
                <main className="main-content">
                    <button onClick={handleBack} className="back-button">
                        <FaArrowLeft /> Back to News
                    </button>
                    <p>Article not found.</p>
                </main>
                <Footer />
            </div>
        );
    }

    return (
        <div>
            <Navbar />
            <main className="main-content">
                <button onClick={handleBack} className="back-button">
                    <FaArrowLeft />
                </button>
                <article>
                    <h1>{article.title}</h1>
                    {article.urlToImage && (
                        <img src={article.urlToImage} alt={article.title} className="article-image" />
                    )}
                    <p><strong>Published on:</strong> {new Date(article.publishedAt).toLocaleDateString()}</p>
                    <p><strong>Source:</strong> {article.source?.name || "Unknown"}</p>
                    <p>{article.content || article.description}</p>
                    <a href={article.url} target="_blank" rel="noopener noreferrer">Read full article</a>
                </article>
            </main>
            <Footer />
        </div>
    );
};

export default ArticleDetailPage;
