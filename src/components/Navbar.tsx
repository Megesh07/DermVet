import React from 'react';
import { Link } from 'react-router-dom';

const Navbar = () => {
  return (
    <nav className="bg-white shadow-md">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="text-xl font-bold text-pink-600">
              SkinFix
            </Link>
          </div>
          
          <div className="flex items-center space-x-4">
            <Link to="/" className="text-gray-700 hover:text-pink-600">
              Home
            </Link>
            <Link to="/dermatologist" className="text-gray-700 hover:text-pink-600">
              Find Doctor
            </Link>
            <Link to="/emart" className="text-gray-700 hover:text-pink-600">
              E-Mart
            </Link>
            <Link to="/routine" className="text-gray-700 hover:text-pink-600">
              Routine
            </Link>
            <Link to="/chat" className="text-gray-700 hover:text-pink-600">
              Chat
            </Link>
            <Link to="/auth" className="px-4 py-2 bg-pink-600 text-white rounded-lg hover:bg-pink-700">
              Login
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar; 