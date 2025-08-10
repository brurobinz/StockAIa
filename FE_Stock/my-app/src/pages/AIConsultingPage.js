import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import Navbar from '../components/common/Navbar';
import styles from '../pages/AIConsultingPage.module.css';
import menuIcon from '../assets/menu.png';
import upArrow from '../assets/up-arrow.png';
import defaultAvatar from '../assets/avatar.png';

const AIConsultingPage = () => {
  const [inputValue, setInputValue] = useState('');
  const [greetingVisible, setGreetingVisible] = useState(true);
  const [messages, setMessages] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [userData, setUserData] = useState({
    username: 'User',
    email: 'user@example.com',
    avatar: defaultAvatar,
    avatarTimestamp: new Date().getTime(),
  });
  const [chatSessions, setChatSessions] = useState([]);
  const [currentSession, setCurrentSession] = useState(null);
  const [firstMessages, setFirstMessages] = useState({});
  const messagesEndRef = useRef(null);
  const typingIntervalRef = useRef(null);
  const typingTimeoutRef = useRef(null);

  const toggleHistorySidebar = () => {
    setIsHistoryOpen(!isHistoryOpen);
  };

  useEffect(() => {
    const searchQuery = localStorage.getItem('searchQuery');
    if (searchQuery) {
      setInputValue(searchQuery);
      handleSend(searchQuery);
      localStorage.removeItem('searchQuery');
    }
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  useEffect(() => {
    return () => {
      if (typingIntervalRef.current) {
        clearInterval(typingIntervalRef.current);
      }
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
      }
    };
  }, []);

  useEffect(() => {
    const fetchUserData = async () => {
      const token = localStorage.getItem('token');
      const storedUserId = localStorage.getItem('userId');
      if (token && storedUserId) {
        try {
          const response = await axios.get(`http://localhost:4000/api/users/${storedUserId}`, {
            headers: {
              Authorization: `Bearer ${token}`,
            },
          });
          const { username, email, avatar } = response.data;
          setUserData({
            username: username || 'User',
            email: email || 'user@example.com',
            avatar: avatar || defaultAvatar,
            avatarTimestamp: new Date().getTime(),
          });
        } catch (error) {
          console.error('Error fetching user data:', error);
        }
      }
    };
    fetchUserData();
  }, []);

  useEffect(() => {
    const fetchChatSessions = async () => {
      const token = localStorage.getItem('token');
      if (token) {
        try {
          const response = await axios.get('http://localhost:4000/api/chat/sessions', {
            headers: {
              Authorization: `Bearer ${token}`,
            },
          });
          setChatSessions(response.data);

          const firstMessagePromises = response.data.map(session =>
            axios.get(`http://localhost:4000/api/chat/sessions/${session._id}/messages`, {
              headers: { Authorization: `Bearer ${token}` },
              params: { limit: 1, sort: 'asc' }
            })
          );

          const firstMessagesResponses = await Promise.all(firstMessagePromises);
          const firstMessagesMap = {};
          firstMessagesResponses.forEach((res, index) => {
            const sessionId = response.data[index]._id;
            firstMessagesMap[sessionId] = res.data.length > 0 ? res.data[0].text : 'New Chat Session';
          });

          setFirstMessages(firstMessagesMap);

          // Retrieve the last selected session from localStorage
          const savedSessionId = localStorage.getItem('currentSession');
          if (savedSessionId && response.data.some(session => session._id === savedSessionId)) {
            setCurrentSession(savedSessionId);
            fetchChatMessages(savedSessionId);
          } else if (response.data.length > 0) {
            // If no session is saved, default to the first session
            setCurrentSession(response.data[0]._id);
            fetchChatMessages(response.data[0]._id);
            localStorage.setItem('currentSession', response.data[0]._id);
          }
        } catch (error) {
          console.error('Error fetching chat sessions or messages:', error);
        }
      }
    };
    fetchChatSessions();
  }, []);

  const fetchChatMessages = async (sessionId) => {
    const token = localStorage.getItem('token');
    if (token && sessionId) {
      try {
        const response = await axios.get(`http://localhost:4000/api/chat/sessions/${sessionId}/messages`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        setMessages(response.data);
      } catch (error) {
        console.error('Error fetching chat messages:', error);
      }
    }
  };

  const createNewChatSession = async () => {
    const token = localStorage.getItem('token');
    if (token) {
      try {
        const response = await axios.post(
          'http://localhost:4000/api/chat/sessions',
          { title: 'New Chat Session' },
          { headers: { Authorization: `Bearer ${token}` } }
        );
        setChatSessions([response.data, ...chatSessions]);
        setCurrentSession(response.data._id);
        setMessages([]);
        setGreetingVisible(true);
        setFirstMessages(prev => ({
          ...prev,
          [response.data._id]: 'New Chat Session'
        }));
        // Save the new session as the current session in localStorage
        localStorage.setItem('currentSession', response.data._id);
      } catch (error) {
        console.error('Error creating new chat session:', error);
      }
    }
  };

  const switchChatSession = (sessionId) => {
    setCurrentSession(sessionId);
    fetchChatMessages(sessionId);
    setMessages([]);
    setGreetingVisible(true);
    // Save the selected session to localStorage
    localStorage.setItem('currentSession', sessionId);
  };

  const handleInputChange = (e) => {
    setInputValue(e.target.value);
  };

  const typeBotMessage = (text) => {
    let index = 0;
    const typingSpeed = 20;
  
    
    const typeCharacter = () => {
      const currentText = text.substring(0, index + 1); 
      setMessages(prevMessages => {
        const newMessages = [...prevMessages];
        const lastMessage = newMessages[newMessages.length - 1];
        if (lastMessage && lastMessage.sender === 'bot' && lastMessage.isTyping) {
          newMessages[newMessages.length - 1] = { ...lastMessage, text: currentText };
        } else {
          newMessages.push({ text: currentText, sender: 'bot', isTyping: true });
        }
        return newMessages;
      });
  
      // Scroll đến cuối mỗi lần nhả chữ
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  
      if (index < text.length - 1) {
        index++;
        typingTimeoutRef.current = setTimeout(typeCharacter, typingSpeed);
      } else {
        setMessages(prevMessages => {
          const newMessages = [...prevMessages];
          const lastMessage = newMessages[newMessages.length - 1];
          if (lastMessage && lastMessage.sender === 'bot') {
            newMessages[newMessages.length - 1] = { ...lastMessage, isTyping: false };
          }
          return newMessages;
        });
        setIsTyping(false);
      }
    };
  
    // Bắt đầu gõ
    typingTimeoutRef.current = setTimeout(typeCharacter, typingSpeed);
  };
  

  const handleSend = async () => {
    if (inputValue.trim() !== '' && currentSession) {
      const userMessage = { text: inputValue, sender: 'user' };
      setMessages((prevMessages) => [...prevMessages, userMessage]);
      setInputValue('');
      setGreetingVisible(false);
      setIsTyping(true);

      try {
        const token = localStorage.getItem('token');

        // Send user's message to the server
        await axios.post(
          `http://localhost:4000/api/chat/sessions/${currentSession}/messages`,
          { text: inputValue, sender: 'user' },
          { headers: { Authorization: `Bearer ${token}` } }
        );

        // Get response from chatbot API
        const aiResponse = await axios.post(
          'http://localhost:5000/chatbot',
          { query: inputValue },
          {
            headers: {
              Authorization: `Bearer ${token}`,
              'Content-Type': 'application/json'
            }
          }
        );

        const botText = aiResponse.data.response;

        // Send bot's message to the server
        await axios.post(
          `http://localhost:4000/api/chat/sessions/${currentSession}/messages`,
          { text: botText, sender: 'bot' },
          { headers: { Authorization: `Bearer ${token}` } }
        );

        // Display bot's response with typing effect
        typeBotMessage(botText);

      } catch (error) {
        console.error('Error handling message:', error);
        setIsTyping(false);
      }
    }
  };

  return (
    <div className={styles.aiConsultingPage}>
      <Navbar />

      {/* History Sidebar */}
      <div className={`${styles.historySidebar} ${isHistoryOpen ? styles.open : ''}`}>
        <h2 className={styles.sidebarTitle}>Conversation History</h2>
        <button className={styles.newChatButton} onClick={createNewChatSession}>New Chat</button>
        <div className={styles.chatSessionList}>
          {chatSessions.map((session) => (
            <div
              key={session._id}
              className={`${styles.chatSessionItem} ${currentSession === session._id ? styles.activeSession : ''}`}
              onClick={() => switchChatSession(session._id)}
            >
              <div className={styles.sessionHeader}>
                <img src={userData.avatar} alt="Avatar" className={styles.sessionAvatar} />
                <div className={styles.sessionInfo}>
                  <span className={styles.chatSessionTitle}>{firstMessages[session._id]}</span>
                  <span className={styles.chatSessionDate}>
                    {new Date(session.createdAt).toLocaleString()}
                  </span>
                </div>
              </div>
            </div>
          ))}
          {chatSessions.length === 0 && (
            <div className={styles.noSessions}>
              No chat history available.
            </div>
          )}
        </div>
      </div>

      {/* Toggle Button for Sidebar */}
      <button className={styles.menuButton} onClick={toggleHistorySidebar}>
        <img src={menuIcon} alt="Menu" className={styles.menuIcon} />
      </button>

      {/* Main Chat Container */}
      <div className={styles.mainContainer}>
        {/* Greeting */}
        {greetingVisible && (
          <div className={styles.greetingContainer}>
            <h1>
              <span className={styles.hello}>Hello </span>
              <span className={styles.user}>{userData.username}</span>
              <span className={styles.exclamation}>!</span>
            </h1>
          </div>
        )}

        {/* Messages Container */}
        <div className={styles.messagesContainer}>
          {messages.map((message, index) => (
            <div
              key={index}
              className={`${styles.messageWrapper} ${message.sender === 'user' ? styles.userMessage : styles.botMessage}`}
            >
              {message.sender === 'bot' && (
                <img src={defaultAvatar} alt="Bot Avatar" className={styles.messageAvatar} />
              )}
              <div className={styles.messageContent}>
                <div className={styles.messageText}>{message.text}</div>
              </div>
              {message.sender === 'user' && (
                <img src={userData.avatar} alt="User Avatar" className={styles.messageAvatar} />
              )}
            </div>
          ))}
          {isTyping && (
            <div className={styles.typingIndicator}>
              <span></span>
              <span></span>
              <span></span>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Chat Input */}
        <div className={styles.chatInputContainer}>
          <textarea
            placeholder="Ask AI Consulting..."
            value={inputValue}
            onChange={handleInputChange}
            onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
            className={styles.chatInput}
            rows={1}
            disabled={isTyping || !currentSession}
          />
          <button
            className={styles.sendButton}
            onClick={handleSend}
            disabled={isTyping || !currentSession}
          >
            <img src={upArrow} alt="Send" className={styles.upArrowIcon} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default AIConsultingPage;
