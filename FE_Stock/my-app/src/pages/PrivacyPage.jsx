import React from 'react';
import './PrivacyPage.css';

const PrivacyPage = () => {
    return (
        <div className="privacy-container">
            <h1 className="privacy-title">Privacy Policy & Terms of Use</h1>

            <section className="privacy-section">
                <h2>1. Introduction</h2>
                <p>
                    Welcome to StockInsight. By accessing and using our services, you agree to be bound by the terms and conditions outlined in this document. This policy governs your use of our platform and explains how we collect, use, and safeguard your personal data.
                </p>
            </section>

            <section className="privacy-section">
                <h2>2. Information We Collect</h2>
                <ul>
                    <li>Personal Information: Name, email, contact details, and account activity.</li>
                    <li>Usage Data: Information about how you interact with the platform.</li>
                    <li>Device Data: Browser type, IP address, and device operating system.</li>
                </ul>
            </section>

            <section className="privacy-section">
                <h2>3. How We Use Your Information</h2>
                <p>
                    The collected data is used to:
                </p>
                <ul>
                    <li>Authenticate your identity and secure your account.</li>
                    <li>Personalize your experience based on your preferences.</li>
                    <li>Provide customer support and improve service quality.</li>
                    <li>Comply with legal obligations.</li>
                </ul>
            </section>

            <section className="privacy-section">
                <h2>4. Data Sharing & Disclosure</h2>
                <p>
                    We do not sell your personal data. However, we may share your information with:
                </p>
                <ul>
                    <li>Service providers who assist in platform operation (e.g., hosting, analytics).</li>
                    <li>Regulatory or legal authorities if required by law.</li>
                </ul>
            </section>

            <section className="privacy-section">
                <h2>5. Your Rights</h2>
                <p>You have the right to:</p>
                <ul>
                    <li>Access, update, or delete your personal data.</li>
                    <li>Withdraw consent for data processing (where applicable).</li>
                    <li>Receive a copy of the data we hold about you.</li>
                </ul>
                <p>
                    To exercise your rights, please contact us via <a href="mailto:support@stockinsight.com">support@stockinsight.com</a>.
                </p>
            </section>

            <section className="privacy-section">
                <h2>6. Data Security</h2>
                <p>
                    We use industry-standard security practices such as encryption, firewalls, and secure authentication protocols to protect your data from unauthorized access.
                </p>
            </section>

            <section className="privacy-section">
                <h2>7. Cookies & Tracking Technologies</h2>
                <p>
                    Our website may use cookies to enhance your experience and analyze usage. You can disable cookies in your browser settings, but some features may not work properly.
                </p>
            </section>

            <section className="privacy-section">
                <h2>8. Children’s Privacy</h2>
                <p>
                    StockInsight is not intended for users under the age of 13. We do not knowingly collect data from children.
                </p>
            </section>

            <section className="privacy-section">
                <h2>9. Changes to This Policy</h2>
                <p>
                    We may update this privacy policy from time to time. All updates will be posted on this page with the revised date. Continued use of our platform indicates your acceptance of the changes.
                </p>
            </section>

            <section className="privacy-section">
                <h2>10. Contact Us</h2>
                <p>
                    If you have any questions or concerns about this policy, please reach out to us:
                </p>
                <ul>
                    <li>Email: <a href="mailto:support@stockinsight.com">support@stockinsight.com</a></li>
                    <li>Phone: +84 123 456 789</li>
                    <li>Address: 123 Finance Street, Hanoi, Vietnam</li>
                </ul>
            </section>
        </div>
    );
};

export default PrivacyPage;
