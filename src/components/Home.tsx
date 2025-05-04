import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Link } from 'react-router-dom';
import { Bot, ShoppingBag, Calendar, ArrowRight, ChevronLeft, ChevronRight, Heart, Instagram, Twitter, Facebook, Youtube, PawPrint } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

const slides = [
  {
    url: "https://plus.unsplash.com/premium_photo-1677165327781-1c0b7c458821?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    title: "Veterinary Skin Health Solutions"
  },
  {
    url: "https://plus.unsplash.com/premium_photo-1677165479692-180fac4c0832?q=80&w=2055&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    title: "Personalized Products Suggestions"
  },
  {
    url: "https://images.unsplash.com/photo-1597214840472-aa1eaf0e1fac?q=80&w=2072&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    title: "Expert Veterinary Dermatology Care"
  },
  {
    url: "https://plus.unsplash.com/premium_photo-1664300840401-4203f174981a?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    title: "AI-Powered Analysis"
  },
  {
    url: "https://images.unsplash.com/photo-1450778869180-41d0601e046e?q=80&w=1972&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    title: "Trusted Solutions for Animal Skin Conditions"
  }
];

const features = [
  {
    icon: Bot,
    title: "Derma Vet AI",
    description: "Get personalized advice from our AI assistant",
    link: "/chat",
    color: "bg-pink-100",
    textColor: "text-pink-600"
  },
  {
    icon: ShoppingBag,
    title: "AnimLuxe",
    description: "Shop for recommended products",
    link: "/emart",
    color: "bg-rose-100",
    textColor: "text-rose-600"
  },
  {
    icon: Calendar,
    title: "Expert Consultation",
    description: "Book appointments with certified veterinarian",
    link: "/dermatologist",
    color: "bg-pink-100",
    textColor: "text-pink-600"
  },
  {
    icon: PawPrint,
    title: "Skin Disease Diagnosis",
    description: "Get AI-powered analysis for pet skin conditions",
    link: "/diagnosis",
    color: "bg-rose-100",
    textColor: "text-rose-600"
  }

];

const containerVariants = {
  hidden: { 
    opacity: 0
  },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.3
    }
  }
};

const itemVariants = {
  hidden: { 
    opacity: 0,
    y: 20
  },
  show: { 
    opacity: 1,
    y: 0,
    transition: {
      type: "spring",
      stiffness: 300,
      damping: 24
    }
  }
};

function Home() {
  const [currentSlide, setCurrentSlide] = useState(0);
  const { isAuthenticated } = useAuth();

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentSlide((prev) => (prev + 1) % slides.length);
    }, 5000);
    return () => clearInterval(timer);
  }, []);

  const nextSlide = () => {
    setCurrentSlide((prev) => (prev + 1) % slides.length);
  };

  const prevSlide = () => {
    setCurrentSlide((prev) => (prev - 1 + slides.length) % slides.length);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#FFF5EE] to-white">
      <div className="relative h-[50vh] md:h-[500px] mb-12 overflow-hidden">
        <AnimatePresence mode="wait">
          <motion.div
            key={currentSlide}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
            className="absolute inset-0"
          >
            <div className="absolute inset-0 bg-black/30 z-10" />
            <img
              src={slides[currentSlide].url}
              alt={slides[currentSlide].title}
              className="w-full h-full object-cover"
            />
            <div className="absolute inset-0 flex items-center justify-center z-20 px-4">
              <motion.h2
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.2 }}
                className="text-3xl md:text-5xl font-bold text-white text-center shadow-lg"
              >
                {slides[currentSlide].title}
              </motion.h2>
            </div>
          </motion.div>
        </AnimatePresence>
        
        <button
          onClick={prevSlide}
          className="absolute left-4 top-1/2 -translate-y-1/2 z-30 bg-white/20 backdrop-blur-sm p-2 rounded-full hover:bg-white/40 transition-all"
        >
          <ChevronLeft className="w-6 h-6 text-white" />
        </button>
        <button
          onClick={nextSlide}
          className="absolute right-4 top-1/2 -translate-y-1/2 z-30 bg-white/20 backdrop-blur-sm p-2 rounded-full hover:bg-white/40 transition-all"
        >
          <ChevronRight className="w-6 h-6 text-white" />
        </button>

        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-30 flex gap-2">
          {slides.map((_, index) => (
            <button
              key={index}
              onClick={() => setCurrentSlide(index)}
              className={`w-2 h-2 rounded-full transition-all ${
                currentSlide === index ? 'bg-white w-6' : 'bg-white/50'
              }`}
            />
          ))}
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="show"
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-8 mb-16"
        >
          {features.map((feature, index) => (
            <motion.div
              key={index}
              variants={itemVariants}
              className="relative group"
            >
              <Link to={feature.link}>
                <motion.div
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className={`${feature.color} rounded-2xl shadow-lg p-6 md:p-8 h-full transition-all duration-300 hover:shadow-xl`}
                >
                  <feature.icon className={`w-10 h-10 md:w-12 md:h-12 ${feature.textColor} mb-4`} />
                  <h3 className="text-xl md:text-2xl font-bold text-gray-900 mb-2 md:mb-3">{feature.title}</h3>
                  <p className="text-sm md:text-base text-gray-600">{feature.description}</p>
                </motion.div>
              </Link>
            </motion.div>
          ))}
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.5 }}
          className="text-center bg-gradient-to-r from-pink-50 to-rose-50 p-6 md:p-12 rounded-3xl shadow-lg mb-12"
        >
          <h2 className="text-2xl md:text-3xl font-bold text-gray-900 mb-4 md:mb-6">
           Revolutionize Pet Skincare with AI!
          </h2>
          <p className="text-sm md:text-base text-gray-600 mb-6 md:mb-8 max-w-2xl mx-auto">
           Join the revolution in pet skin health with Derma Vet AI!
          </p>
          <Link to={isAuthenticated ? "/chat" : "/auth"}>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="bg-gradient-to-r from-pink-500 to-rose-500 text-white px-8 md:px-12 py-3 md:py-4 rounded-full font-semibold text-base md:text-lg shadow-lg hover:shadow-xl transition-all duration-300"
            >
              {isAuthenticated ? 'Start Chat' : 'Get Started Now'}
            </motion.button>
          </Link>
        </motion.div>
      </div>

      <footer className="bg-gray-900 text-white mt-24">
        <div className="max-w-7xl mx-auto px-4 py-8 md:py-12">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8">
            <div>
              <h3 className="text-2xl font-bold mb-4 bg-gradient-to-r from-pink-500 to-rose-500 bg-clip-text text-transparent">
                Derma Vet AI
              </h3>
              <p className="text-gray-400 mb-4 text-sm md:text-base">
                Enhancing animal skincare with the power of AI and veterinary excellence.Enhancing animal skincare with the power of AI and veterinary excellence.
              </p>
              <div className="flex space-x-4">
                <a href="#" className="text-gray-400 hover:text-white transition-colors">
                  <Instagram className="w-5 h-5 md:w-6 md:h-6" />
                </a>
                <a href="#" className="text-gray-400 hover:text-white transition-colors">
                  <Twitter className="w-5 h-5 md:w-6 md:h-6" />
                </a>
                <a href="#" className="text-gray-400 hover:text-white transition-colors">
                  <Facebook className="w-5 h-5 md:w-6 md:h-6" />
                </a>
                <a href="#" className="text-gray-400 hover:text-white transition-colors">
                  <Youtube className="w-5 h-5 md:w-6 md:h-6" />
                </a>
              </div>
            </div>
            
            <div>
              <h4 className="text-lg font-semibold mb-4">Quick Links</h4>
              <ul className="space-y-2">
                <li>
                  <Link to="/chat" className="text-gray-400 hover:text-white transition-colors text-sm md:text-base">
                    Derma Vet AI
                  </Link>
                </li>
                <li>
                  <Link to="/emart" className="text-gray-400 hover:text-white transition-colors text-sm md:text-base">
                    AnimLuxe
                  </Link>
                </li>
                <li>
                  <Link to="/dermatologist" className="text-gray-400 hover:text-white transition-colors text-sm md:text-base">
                    Book Consultation
                  </Link>
                </li>
              </ul>
            </div>
            
            <div>
              <h4 className="text-lg font-semibold mb-4">Support</h4>
              <ul className="space-y-2">
                <li>
                  <a href="#" className="text-gray-400 hover:text-white transition-colors text-sm md:text-base">
                    Help Center
                  </a>
                </li>
                <li>
                  <a href="#" className="text-gray-400 hover:text-white transition-colors text-sm md:text-base">
                    Privacy Policy
                  </a>
                </li>
                <li>
                  <a href="#" className="text-gray-400 hover:text-white transition-colors text-sm md:text-base">
                    Terms of Service
                  </a>
                </li>
                <li>
                  <a href="#" className="text-gray-400 hover:text-white transition-colors text-sm md:text-base">
                    Contact Us
                  </a>
                </li>
              </ul>
            </div>
            
            <div>
              <h4 className="text-lg font-semibold mb-4">Newsletter</h4>
              <p className="text-gray-400 mb-4 text-sm md:text-base">
                Stay updated with the latest skincare tips and product launches.
              </p>
              <form className="space-y-2">
                <input
                  type="email"
                  placeholder="Enter your email"
                  className="w-full px-4 py-2 rounded-lg bg-gray-800 border border-gray-700 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-pink-500 text-sm md:text-base"
                />
                <button
                  type="submit"
                  className="w-full px-4 py-2 bg-gradient-to-r from-pink-500 to-rose-500 text-white rounded-lg hover:from-pink-600 hover:to-rose-600 transition-colors text-sm md:text-base"
                >
                  Subscribe
                </button>
              </form>
            </div>
          </div>
          
          <div className="mt-8 md:mt-12 pt-6 md:pt-8 border-t border-gray-800 text-center text-gray-400">
            <p className="text-sm md:text-base">© 2025 Derma Vet AI. All rights reserved.</p>
            <p className="mt-2 flex items-center justify-center text-sm md:text-base">
              Made with <Heart className="w-4 h-4 mx-1 text-pink-500" /> by Derma Vet AI Team
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default Home;