import React, { useEffect, useState, useCallback } from 'react';
import AnimatedBackground from './components/AnimatedBackground';
import AuthPage from './components/AuthPage';
import ProfileSetupPage from './components/ProfileSetupPage';
import AgentSelectionPage from './components/AgentSelectionPage';
import ChatPage from './components/ChatPage';
import DashboardPage from './components/DashboardPage';
import SettingsPage from './components/SettingsPage';
import AboutPage from './components/AboutPage';
import GithubAnalysisPage from './components/GithubAnalysisPage';
import FileUploadModal from './components/FileUploadModal';
import LoadingOverlay from './components/LoadingOverlay';
import Notification from './components/Notification';
import { getSystemStatus } from './utils';

const PAGES = {
  AUTH: 1,
  PROFILE: 2,
  AGENT: 3,
  CHAT: 4,
  DASHBOARD: 5,
  SETTINGS: 6,
  ABOUT: 7,
  GITHUB: 8,
};

function App() {
  const [currentPage, setCurrentPage] = useState(PAGES.AUTH);
  const [currentUser, setCurrentUser] = useState(null);
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [showFileModal, setShowFileModal] = useState(false);
  const [loading, setLoading] = useState({ show: false, message: '' });
  const [notification, setNotification] = useState(null);

  useEffect(() => {
    const isLoggedIn = localStorage.getItem('isLoggedIn');
    const profileData = localStorage.getItem('profileData');
    if (isLoggedIn === 'true') {
      setCurrentUser({ email: localStorage.getItem('userEmail') || 'user@gmail.com' });
      setCurrentPage(profileData ? PAGES.CHAT : PAGES.PROFILE);
    } else {
      setCurrentPage(PAGES.AUTH);
    }
    getSystemStatus();
  }, []);

  const showNotificationMessage = useCallback((message, type) => {
    setNotification({ message, type });
    setTimeout(() => setNotification(null), 5000);
  }, []);

  const navigateTo = useCallback((page) => {
    setCurrentPage(page);
  }, []);

  return (
    <div className="relative min-h-screen w-full">
      <AnimatedBackground />
      {notification && (
        <Notification message={notification.message} type={notification.type} />
      )}
      {loading.show && <LoadingOverlay message={loading.message} />}
      {showFileModal && (
        <FileUploadModal
          onClose={() => setShowFileModal(false)}
          showNotification={showNotificationMessage}
          setLoading={setLoading}
        />
      )}
      {{
        [PAGES.AUTH]: (
          <AuthPage
            onLoginSuccess={(user) => {
              setCurrentUser(user);
              navigateTo(PAGES.PROFILE);
            }}
            showNotification={showNotificationMessage}
          />
        ),
        [PAGES.PROFILE]: (
          <ProfileSetupPage
            onProfileComplete={() => navigateTo(PAGES.AGENT)}
            showNotification={showNotificationMessage}
          />
        ),
        [PAGES.AGENT]: (
          <AgentSelectionPage
            onAgentSelect={(agent) => setSelectedAgent(agent)}
            selectedAgent={selectedAgent}
            onContinue={() => navigateTo(PAGES.CHAT)}
            showNotification={showNotificationMessage}
          />
        ),
        [PAGES.CHAT]: (
          <ChatPage
            currentUser={currentUser}
            selectedAgent={selectedAgent}
            setShowFileModal={setShowFileModal}
            navigateTo={navigateTo}
            PAGES={PAGES}
            showNotification={showNotificationMessage}
            setLoading={setLoading}
          />
        ),
        [PAGES.DASHBOARD]: <DashboardPage onBack={() => navigateTo(PAGES.CHAT)} />, 
        [PAGES.SETTINGS]: <SettingsPage onBack={() => navigateTo(PAGES.CHAT)} />, 
        [PAGES.ABOUT]: <AboutPage onBack={() => navigateTo(PAGES.CHAT)} />, 
        [PAGES.GITHUB]: <GithubAnalysisPage onBack={() => navigateTo(PAGES.CHAT)} showNotification={showNotificationMessage} setLoading={setLoading} />, 
      }[currentPage]}
    </div>
  );
}

export default App;