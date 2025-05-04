import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { motion } from 'framer-motion';
import { User, Lock, Mail, ArrowRight } from 'lucide-react';

// Define user credentials (hidden from UI)
const TEST_USERS = {
  clients: [
    { email: 'client1@test.com', password: 'client123', name: 'John Doe' },
    { email: 'client2@test.com', password: 'client456', name: 'Jane Smith' },
    { email: 'client3@test.com', password: 'client789', name: 'Mike Johnson' }
  ],
  dermatologists: [
    { email: 'derm1@test.com', password: 'derm123', name: 'Dr. Sarah Wilson' },
    { email: 'derm2@test.com', password: 'derm456', name: 'Dr. James Brown' },
    { email: 'derm3@test.com', password: 'derm789', name: 'Dr. Emily Davis' }
  ]
};

export default function Auth() {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState('client');
  const [error, setError] = useState('');
  const { login } = useAuth();
  const navigate = useNavigate();

  const validateUser = (email: string, password: string, role: string) => {
    const userList = TEST_USERS[role === 'client' ? 'clients' : 'dermatologists'];
    if (!userList) {
      console.error('Invalid role specified');
      return null;
    }
    return userList.find(user => user.email === email && user.password === password);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    try {
      // For login, try both client and dermatologist roles
      if (isLogin) {
        const clientUser = validateUser(email, password, 'client');
        const dermUser = validateUser(email, password, 'dermatologist');
        
        if (clientUser) {
          await login(email, password, 'client');
          navigate('/find-dermatologist');
        } else if (dermUser) {
          await login(email, password, 'dermatologist');
          navigate('/client-requests');
        } else {
          setError('Invalid email or password');
        }
      } else {
        // For signup, use selected role
        const user = validateUser(email, password, role);
        if (user) {
          setError('User already exists');
        } else {
          await login(email, password, role);
          navigate(role === 'client' ? '/find-dermatologist' : '/client-requests');
        }
      }
    } catch (error) {
      console.error('Auth error:', error);
      setError('Authentication failed. Please try again.');
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-pink-50 to-[#FFF5EE] py-12 px-4 sm:px-6 lg:px-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="max-w-md w-full"
      >
        <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
          <div className="px-8 py-6">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold bg-gradient-to-r from-pink-500 to-[#D2691E] bg-clip-text text-transparent">
                {isLogin ? 'Welcome Back!' : 'Create Account'}
              </h2>
              <p className="mt-2 text-gray-600">
                {isLogin 
                  ? 'Please sign in to your account' 
                  : 'Join us to get started with your skincare journey'}
              </p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="space-y-4">
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-5 w-5" />
                  <input
                    type="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="w-full pl-10 pr-4 py-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-pink-500 focus:border-transparent transition-colors bg-gray-50"
                    placeholder="Email address"
                  />
                </div>

                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-5 w-5" />
                  <input
                    type="password"
                    required
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="w-full pl-10 pr-4 py-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-pink-500 focus:border-transparent transition-colors bg-gray-50"
                    placeholder="Password"
                  />
                </div>

                {/* Only show role selection in sign-up mode */}
                {!isLogin && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="flex flex-col space-y-2 p-4 bg-gray-50 rounded-lg"
                  >
                    <p className="text-sm font-medium text-gray-700 mb-2">I am a:</p>
                    <div className="flex space-x-4">
                      <label className="flex items-center space-x-2 cursor-pointer group">
                        <input
                          type="radio"
                          value="client"
                          checked={role === 'client'}
                          onChange={(e) => setRole(e.target.value)}
                          className="form-radio text-pink-600 focus:ring-pink-500"
                        />
                        <span className="text-gray-700 group-hover:text-pink-600 transition-colors">
                          Client
                        </span>
                      </label>
                      <label className="flex items-center space-x-2 cursor-pointer group">
                        <input
                          type="radio"
                          value="dermatologist"
                          checked={role === 'dermatologist'}
                          onChange={(e) => setRole(e.target.value)}
                          className="form-radio text-pink-600 focus:ring-pink-500"
                        />
                        <span className="text-gray-700 group-hover:text-pink-600 transition-colors">
                          Dermatologist
                        </span>
                      </label>
                    </div>
                  </motion.div>
                )}

                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="bg-red-50 border border-red-200 text-red-600 px-4 py-3 rounded-lg text-sm"
                  >
                    {error}
                  </motion.div>
                )}
              </div>

              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                type="submit"
                className="w-full py-3 px-4 flex items-center justify-center space-x-2 bg-gradient-to-r from-pink-500 to-[#D2691E] text-white rounded-lg hover:opacity-90 transition-opacity focus:outline-none focus:ring-2 focus:ring-pink-500 focus:ring-offset-2"
              >
                <span>{isLogin ? 'Sign In' : 'Create Account'}</span>
                <ArrowRight className="h-4 w-4" />
              </motion.button>

              <div className="text-center">
                <button
                  type="button"
                  onClick={() => {
                    setIsLogin(!isLogin);
                    setError('');
                  }}
                  className="text-sm text-pink-600 hover:text-pink-700 focus:outline-none"
                >
                  {isLogin 
                    ? "Don't have an account? Sign up" 
                    : 'Already have an account? Sign in'}
                </button>
              </div>
            </form>
          </div>

          <div className="px-8 py-4 bg-gray-50 border-t border-gray-100">
            <p className="text-xs text-center text-gray-600">
              By continuing, you agree to our Terms of Service and Privacy Policy
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
}