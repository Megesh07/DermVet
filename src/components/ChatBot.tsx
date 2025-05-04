import React, { useState, useRef, useEffect } from 'react';
import { Send, Sparkles, Camera, Upload, X, Trash2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import config from "@/config";  // ✅ Use alias instead of `../config`
import axios from "axios";

interface Message {
  type: 'user' | 'bot' | 'assistant';
  content: string;
  image?: string;
  imagePath?: string;
  imageAnalysis?: {
    condition: string;
    confidence: string;
    imagePath: string;
  } | null;
}

export default function ChatBot() {
  const [messages, setMessages] = useState<Message[]>(() => {
    const savedMessages = localStorage.getItem('chatMessages');
    return savedMessages ? JSON.parse(savedMessages) : [{
      type: 'bot',
      content: 'Hello! I\'m your Derma Vet AI . I can help with your pet\'s skin concerns and recommend personalized solutions. You can send me messages or share photos of your pet\'s skin concerns. How can I assist you today?'
    }];
  });
  
  const [input, setInput] = useState(''); 
  const [isInitial, setIsInitial] = useState(true);
  const [showCamera, setShowCamera] = useState(false);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [showAlert, setShowAlert] = useState(false);
  const [alertMessage, setAlertMessage] = useState('');
  const [isCameraLoading, setIsCameraLoading] = useState(false);
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
    localStorage.setItem('chatMessages', JSON.stringify(messages));
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() && !selectedImage) return;
  
    setIsInitial(false);
    
    setMessages(prev => [...prev, { 
      type: 'user', 
      content: input, 
      image: selectedImage || undefined 
    }]);
  
    setInput('');
    setSelectedImage(null);
    setIsLoading(true);
  
    const formData = new FormData();
    if (input) formData.append('message', input);
    if (selectedImage) {
      const base64Response = await fetch(selectedImage);
      const blob = await base64Response.blob();
      formData.append('image', blob, 'image.jpg');
    }
  
    try {
      const response = await axios.post(`${config.API_URL}/api/chat`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
  
      if (response.data.status === 'success') {
        setMessages(prev => [...prev, {
          type: 'assistant',
          content: response.data.response,
          imagePath: response.data.imagePath
        }]);
      } else {
        setError(response.data.message || 'An error occurred');
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      setError('Failed to send message');
    } finally {
      setIsLoading(false);
    }
  };
  
  const getAvailableCameras = async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(device => device.kind === 'videoinput');
      setCameras(videoDevices);
      
      // Select the first camera by default
      if (videoDevices.length > 0 && !selectedCameraId) {
        setSelectedCameraId(videoDevices[0].deviceId);
      }
    } catch (err) {
      console.error('Error getting cameras:', err);
    }
  };

  const startCamera = async () => {
    setIsCameraLoading(true);
    setCameras([]);
    
    try {
      // First get available cameras
      await getAvailableCameras();
      setShowCamera(true);

      // Set up camera constraints
      const constraints = {
        video: {
          deviceId: selectedCameraId ? { exact: selectedCameraId } : undefined,
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false // Explicitly disable audio
      };
      
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await new Promise<void>((resolve) => {
          if (videoRef.current) {
            videoRef.current.onloadedmetadata = () => {
              videoRef.current?.play().catch(console.error);
              resolve();
            };
          }
        });
        setStream(stream);
        setIsCameraLoading(false);
      }
    } catch (err) {
      console.error('Error accessing camera:', err);
      setAlertMessage('Unable to access webcam. Please ensure you have a working camera connected to your PC.');
      setShowAlert(true);
      setIsCameraLoading(false);
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => {
        track.stop();
      });
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      setStream(null);
    }
    setShowCamera(false);
    setIsCameraLoading(false);
  };

  const capturePhoto = () => {
    if (videoRef.current) {
      const canvas = document.createElement('canvas');
      const video = videoRef.current;
      const aspectRatio = video.videoWidth / video.videoHeight;
      
      canvas.width = 1280; // Set a reasonable max width
      canvas.height = canvas.width / aspectRatio;
      
      const ctx = canvas.getContext('2d');
      ctx?.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      const image = canvas.toDataURL('image/jpeg', 0.8); // Compress to 80% quality
      setSelectedImage(image);
      stopCamera();
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // Validate file type
      if (!file.type.startsWith('image/')) {
        alert('Please upload an image file');
        return;
      }

      // Validate file size (e.g., max 5MB)
      const maxSize = 5 * 1024 * 1024; // 5MB in bytes
      if (file.size > maxSize) {
        alert('File size should be less than 5MB');
        return;
      }

      const reader = new FileReader();
      reader.onloadend = () => {
        setSelectedImage(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const clearChat = () => {
    setMessages([]);
    // You might also want to reset any other chat-related state here
  };

  const LoadingDots = () => {
    return (
      <div className="flex space-x-2 p-4 bg-[#FFE4E1] rounded-2xl max-w-[85%] md:max-w-[75%]">
        <motion.div
          className="w-2 h-2 bg-[#8B4513] rounded-full"
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ duration: 0.5, repeat: Infinity, repeatDelay: 0.2 }}
        />
        <motion.div
          className="w-2 h-2 bg-[#8B4513] rounded-full"
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ duration: 0.5, repeat: Infinity, repeatDelay: 0.2, delay: 0.2 }}
        />
        <motion.div
          className="w-2 h-2 bg-[#8B4513] rounded-full"
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ duration: 0.5, repeat: Infinity, repeatDelay: 0.2, delay: 0.4 }}
        />
      </div>
    );
  };

  const switchCamera = async (deviceId: string) => {  
    setSelectedCameraId(deviceId);
    setIsCameraLoading(true);
  
    try {
      if (!deviceId) {
        console.error("No device ID found.");
        return;
      }
  
      // Stop existing camera stream if any
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
  
      // ✅ Fix: Ensure function is async before using await
      const newStream = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: { exact: deviceId },
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      });
  
      if (videoRef.current) {
        videoRef.current.srcObject = newStream;
        setStream(newStream);
      }
    } catch (err) {
      console.error("Error switching camera:", err);
      setAlertMessage("Failed to switch camera.");
      setShowAlert(true);
    } finally {
      setIsCameraLoading(false);
    }
  };
  
  
  return (
    <div className="h-screen flex flex-col bg-[#FFF5EE]">
      <AnimatePresence mode="wait">
        {isInitial ? (
          <motion.div
            key="initial"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="flex-1 flex flex-col items-center justify-center relative px-4 py-12 mb-8"
          >
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="absolute top-0 left-1/2 transform -translate-x-1/2 w-64 h-64 bg-gradient-to-r from-pink-200 to-[#F5D0C5] rounded-full filter blur-3xl opacity-50 z-0"
            />
            <motion.div
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.3 }}
              className="text-center mb-12 relative z-10"
            >
              <h1 className="text-3xl md:text-5xl font-bold bg-gradient-to-r from-pink-500 to-[#D2691E] text-transparent bg-clip-text mb-8 leading-normal md:leading-relaxed">
                How can I help you today?
              </h1>
              <motion.p 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4 }}
                className="text-lg md:text-xl text-[#8B4513] mt-6"
              >
                Got concerns about your pet's skin? Let AI help!
              </motion.p>
            </motion.div>
            <form onSubmit={handleSubmit} className="w-full max-w-2xl relative z-10 mt-8">
              <div className="relative group">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask about your pet's skin concerns..."
                  className="w-full p-4 md:p-6 text-base md:text-lg border-2 border-pink-200 rounded-2xl focus:outline-none focus:border-pink-500 transition-all duration-300 shadow-lg group-hover:shadow-xl bg-white/80 backdrop-blur-sm"
                />
                <button
                  type="submit"
                  className="absolute right-4 top-1/2 -translate-y-1/2 z-30 bg-gradient-to-r from-pink-500 to-[#D2691E] text-white p-3 rounded-xl transition-all duration-300 hover:scale-105 active:scale-95 flex items-center gap-2"
                >
                  <Sparkles className="w-5 h-5" />
                  <span className="hidden md:inline">Ask Derma Vet AI</span>
                </button>
              </div>
            </form>
          </motion.div>
        ) : (
          <motion.div
            key="chat"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex-1 flex flex-col relative h-full"
          >
            <div className="flex-1 overflow-y-auto pb-[80px]">
              <div className="py-6 space-y-6 px-4 md:px-8">
                {messages.map((message, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                    className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[85%] md:max-w-[75%] rounded-2xl p-3 md:p-4 ${
                        message.type === 'user'
                          ? 'bg-gradient-to-r from-pink-500 to-[#D2691E] text-white'
                          : 'bg-[#FFE4E1] text-[#8B4513]'
                      }`}
                    >
                      {message.image && (
                        <img
                          src={message.image}
                          alt="Uploaded"
                          className="max-w-full rounded-lg mb-2"
                        />
                      )}
                      {message.content}
                    </div>
                  </motion.div>
                ))}
                {isLoading && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex justify-start"
                  >
                    <LoadingDots />
                  </motion.div>
                )}
                <div ref={messagesEndRef} />
              </div>
            </div>

            {showCamera && (
              <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  className="bg-[#FFF5EE] rounded-2xl overflow-hidden max-w-2xl w-full shadow-2xl"
                >
                  <div className="p-4 border-b border-pink-100 flex justify-between items-center">
                    <div className="flex items-center gap-4">
                      <h3 className="text-lg font-semibold text-[#8B4513]">Take a Photo</h3>
                      {cameras.length > 1 && (
                        <select
                          value={selectedCameraId}
                          onChange={(e) => switchCamera(e.target.value)}
                          className="text-sm border border-pink-200 rounded-lg px-2 py-1 bg-white"
                        >
                          {cameras.map((camera) => (
                            <option key={camera.deviceId} value={camera.deviceId}>
                              {camera.label || `Camera ${cameras.indexOf(camera) + 1}`}
                            </option>
                          ))}
                        </select>
                      )}
                    </div>
                    <button 
                      onClick={stopCamera} 
                      className="p-2 hover:bg-pink-50 rounded-full transition-colors"
                    >
                      <X className="w-6 h-6 text-pink-500" />
                    </button>
                  </div>
                  
                  <div className="relative bg-black" style={{ aspectRatio: '16/9' }}>
                    {isCameraLoading && (
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="animate-spin rounded-full h-12 w-12 border-4 border-pink-500 border-t-transparent"></div>
                      </div>
                    )}
                    
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      muted
                      className={`w-full h-full object-cover ${isCameraLoading ? 'opacity-0' : 'opacity-100'} transition-opacity duration-300`}
                    />
                    
                    <div className="absolute inset-x-0 bottom-0 p-4 bg-gradient-to-t from-black/50 to-transparent">
                      <button
                        onClick={capturePhoto}
                        disabled={isCameraLoading}
                        className="w-full bg-gradient-to-r from-pink-500 to-[#D2691E] text-white py-3 rounded-xl hover:shadow-lg transition-all duration-300 flex items-center justify-center gap-2 disabled:opacity-50"
                      >
                        <Camera className="w-5 h-5" />
                        <span>Capture Photo</span>
                      </button>
                    </div>
                  </div>

                  <div className="p-4 bg-pink-50">
                    <p className="text-sm text-[#8B4513] text-center">
                      Position your pet's affected area clearly in the frame
                    </p>
                  </div>
                </motion.div>
              </div>
            )}

            <div className="fixed bottom-0 left-0 right-0 bg-white/80 backdrop-blur-sm border-t border-pink-200 shadow-lg z-40">
              <form onSubmit={handleSubmit} className="max-w-5xl mx-auto p-4">
                {selectedImage && (
                  <div className="absolute bottom-full mb-4 left-4">
                    <div className="relative inline-block">
                      <img src={selectedImage} alt="Selected" className="h-20 rounded-lg" />
                      <button
                        onClick={() => setSelectedImage(null)}
                        className="absolute -top-2 -right-2 bg-pink-500 text-white rounded-full p-1"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                )}
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={startCamera}
                    className="p-3 text-[#8B4513] hover:text-pink-600 transition-colors"
                  >
                    <Camera className="w-6 h-6" />
                  </button>
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="p-3 text-[#8B4513] hover:text-pink-600 transition-colors"
                  >
                    <Upload className="w-6 h-6" />
                  </button>
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileUpload}
                    accept="image/*"
                    className="hidden"
                  />
                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Type your message..."
                    className="flex-1 p-3 border-2 border-pink-200 rounded-xl focus:outline-none focus:border-pink-500 transition-all duration-300"
                  />
                  <motion.button
                    type="button"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={clearChat}
                    className="bg-pink-100 text-pink-600 p-3 rounded-xl hover:bg-pink-200 transition-all duration-300"
                  >
                    <Trash2 className="w-6 h-6" />
                  </motion.button>
                  <button
                    type="submit"
                    className="bg-gradient-to-r from-pink-500 to-[#D2691E] text-white p-3 rounded-xl hover:shadow-lg transition-all duration-300"
                  >
                    <Send className="w-6 h-6" />
                  </button>
                </div>
              </form>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}