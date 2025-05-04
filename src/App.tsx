import React, { useState, useRef, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, Navigate } from 'react-router-dom';
import { Bot, ShoppingBag, Calendar, Menu, X, LogIn, User, Settings as SettingsIcon, LogOut } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import ChatBot from './components/ChatBot';
import EMart from './components/EMart';
import DermatologistBooking from './components/DermatologistBooking';
import Home from './components/Home';
import Auth from './components/Auth';
import Dashboard from './components/Dashboard';
import SettingsPage from './components/Settings';
import { AuthProvider, useAuth } from './context/AuthContext';
import RequestCount from './components/RequestCount';
import ClientRequests from './components/ClientRequests';

function App() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  return (
    <AuthProvider>
      <Router>
        <div className="min-h-screen w-screen overflow-x-hidden bg-gray-50 flex flex-col">
          <nav className="bg-white shadow-lg fixed w-full top-0 z-50">
            <div className="w-full">
              <div className="flex justify-between h-16 px-4">
                <div className="flex items-center flex-shrink-0">
                  <Link to="/" className="flex items-center">
                    <motion.div
                      transition={{ type: "spring", stiffness: 700 }}
                    >
                      <img 
                        src="https://i.ibb.co/fGF30Vjz/1.png"
                        className="h-16 w-auto object-contain"
                      />
                    </motion.div>
                  </Link>
                </div>

                <div className="flex items-center lg:hidden">
                  <button
                    onClick={() => setIsMenuOpen(!isMenuOpen)}
                    className="inline-flex items-center justify-center p-2 text-gray-600 hover:text-gray-900"
                  >
                    {isMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
                  </button>
                </div>

                <div className="hidden lg:flex lg:items-center lg:justify-end lg:flex-1">
                  <NavLinks />
                </div>
              </div>
            </div>

            <AnimatePresence>
              {isMenuOpen && (
                <motion.div 
                  className="lg:hidden w-full"
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                >
                  <div className="w-full bg-white shadow-lg">
                    <NavLinks mobile setIsMenuOpen={setIsMenuOpen} />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </nav>

          <main className="flex-1 w-full pt-16">
            <div className="w-full h-full">
              <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/chat" element={<PrivateRoute><ChatBot /></PrivateRoute>} />
                <Route path="/emart" element={<PrivateRoute><EMart /></PrivateRoute>} />
                <Route path="/dermatologist" element={<PrivateRoute><DermatologistBooking /></PrivateRoute>} />
                <Route path="/dashboard" element={<PrivateRoute><Dashboard /></PrivateRoute>} />
                <Route path="/auth" element={<Auth />} />
                <Route path="/dermatologist-login" element={<DermatologistAuth />} />
                <Route path="/settings" element={<SettingsPage />} />
                <Route 
                  path="/find-dermatologist" 
                  element={
                    <ProtectedRoute>
                      <DermatologistBooking />
                    </ProtectedRoute>
                  } 
                />
                <Route 
                  path="/client-requests" 
                  element={
                    <ProtectedRoute>
                      <RequestCount />
                    </ProtectedRoute>
                  } 
                />
                <Route path="/client-requests" element={<ClientRequests />} />
              </Routes>
            </div>
          </main>
        </div>
      </Router>
    </AuthProvider>
  );
}

function PrivateRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated } = useAuth();
  return isAuthenticated ? <>{children}</> : <Navigate to="/auth" />;
}

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated } = useAuth();
  return isAuthenticated ? <>{children}</> : <Navigate to="/auth" />;
}

function NavLinks({ mobile, setIsMenuOpen }: { mobile?: boolean; setIsMenuOpen?: (value: boolean) => void }) {
  const { isAuthenticated, logout } = useAuth();
  const userMenuRef = useRef<HTMLDivElement>(null);
  const [isMenuOpen, setLocalMenuOpen] = useState(false);
  
  const handleMenuClose = () => {
    setLocalMenuOpen(false);
    setIsMenuOpen?.(false);
  };

  const handleMenuToggle = () => setLocalMenuOpen(!isMenuOpen);
  
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (userMenuRef.current && !userMenuRef.current.contains(event.target as Node)) {
        handleMenuClose();
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [setIsMenuOpen]);

  const links = [
    { to: "/chat", icon: Bot, text: "Derma Vet AI" },
    { to: "/emart", icon: ShoppingBag, text: "AnimLuxe" },
    { to: "/dermatologist", icon: Calendar, text: "Consultation" },
  ];

  const baseClasses = mobile
    ? "flex items-center w-full px-2 py-2 text-base font-medium"
    : "flex items-center px-4 py-2 text-sm font-medium whitespace-nowrap"; // Increased horizontal padding

  const activeClasses = mobile
    ? "bg-pink-100 text-pink-900"
    : "text-pink-900";

  const inactiveClasses = mobile
    ? "text-gray-600"
    : "text-gray-600";

  return (
    <div className={`${mobile ? 'w-full' : 'flex items-center space-x-2'}`}> {/* Added space between items */}
      {links.map(({ to, icon: Icon, text }) => (
        <motion.div
          key={to}
          className={mobile ? "w-full" : ""}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <Link
            to={to}
            className={`${baseClasses} ${window.location.pathname === to ? activeClasses : inactiveClasses}`}
            onClick={handleMenuClose}
          >
            <Icon className="w-5 h-5 mr-2" />
            <span className="flex-1">{text}</span>
          </Link>
        </motion.div>
      ))}
      
      {!isAuthenticated ? (
        <motion.div
          className={mobile ? "w-full" : ""}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <Link
            to="/auth"
            className={`${baseClasses} ${inactiveClasses}`}
            onClick={handleMenuClose}
          >
            <LogIn className="w-5 h-5 mr-2" />
            <span className="flex-1">Login</span>
          </Link>
        </motion.div>
      ) : (
        <div className={`${mobile ? "w-full" : "relative"}`} ref={userMenuRef}>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className={`${baseClasses} ${inactiveClasses} w-full`}
            onClick={handleMenuToggle}
          >
            <User className="w-5 h-5 mr-2" />
            <span className="flex-1">Profile</span>
          </motion.button>
          
          <AnimatePresence>
            {isMenuOpen && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                className={`${mobile ? "w-full" : "absolute right-0 w-48"} bg-white shadow-lg py-1 z-50`}
              >
                <Link
                  to="/dashboard"
                  className="flex items-center w-full px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                  onClick={handleMenuClose}
                >
                  <User className="w-4 h-4 mr-2" />
                  <span className="flex-1">My Dashboard</span>
                </Link>
                <Link
                  to="/settings"
                  className="flex items-center w-full px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                  onClick={handleMenuClose}
                >
                  <SettingsIcon className="w-4 h-4 mr-2" />
                  <span className="flex-1">Settings</span>
                </Link>
                <button
                  onClick={() => {
                    logout();
                    handleMenuClose();
                  }}
                  className="flex items-center w-full px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                >
                  <LogOut className="w-4 h-4 mr-2" />
                  <span className="flex-1">Logout</span>
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
}

function DermatologistAuth() {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    qualification: '',
    experience: '',
    specialization: '',
    consultationFee: ''
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Handle authentication logic here
    console.log('Form submitted:', formData);
  };

  return (
    <div className="max-w-md mx-auto px-4 py-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="bg-white rounded-xl shadow-xl overflow-hidden"
      >
        <div className="p-8">
          <h2 className="text-2xl font-bold text-center text-gray-900 mb-8">
            {isLogin ? 'Dermatologist Login' : 'Register as Dermatologist'}
          </h2>

          <form onSubmit={handleSubmit} className="space-y-6">
            {!isLogin && (
              <>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Full Name
                  </label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    className="w-full p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-pink-300"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Qualification
                  </label>
                  <input
                    type="text"
                    value={formData.qualification}
                    onChange={(e) => setFormData({ ...formData, qualification: e.target.value })}
                    className="w-full p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-pink-300"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Years of Experience
                  </label>
                  <input
                    type="number"
                    value={formData.experience}
                    onChange={(e) => setFormData({ ...formData, experience: e.target.value })}
                    className="w-full p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-pink-300"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Specialization
                  </label>
                  <select
                    value={formData.specialization}
                    onChange={(e) => setFormData({ ...formData, specialization: e.target.value })}
                    className="w-full p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-pink-300"
                    required
                  >
                    <option value="">Select Specialization</option>
                    <option value="general">General Dermatology</option>
                    <option value="cosmetic">Cosmetic Dermatology</option>
                    <option value="pediatric">Pediatric Dermatology</option>
                    <option value="surgical">Surgical Dermatology</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Consultation Fee (USD)
                  </label>
                  <input
                    type="number"
                    value={formData.consultationFee}
                    onChange={(e) => setFormData({ ...formData, consultationFee: e.target.value })}
                    className="w-full p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-pink-300"
                    required
                  />
                </div>
              </>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Email Address
              </label>
              <input
                type="email"
                value={formData.email}
                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                className="w-full p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-pink-300"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Password
              </label>
              <input
                type="password"
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                className="w-full p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-pink-300"
                required
              />
            </div>

            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="w-full bg-gradient-to-r from-pink-500 to-rose-500 text-white py-3 rounded-lg font-semibold hover:from-pink-600 hover:to-rose-600 transition-colors"
              type="submit"
            >
              {isLogin ? 'Sign In' : 'Create Account'}
            </motion.button>
          </form>

          <div className="mt-6 text-center">
            <button
              onClick={() => setIsLogin(!isLogin)}
              className="text-pink-600 hover:text-pink-700 font-medium"
            >
              {isLogin ? "Don't have an account? Register" : 'Already have an account? Sign in'}
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );
}

export default App;