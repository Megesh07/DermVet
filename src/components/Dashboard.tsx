import React from 'react';
import { motion } from 'framer-motion';
import { User, Calendar, Settings, Activity, Bell } from 'lucide-react';

interface PetProfile {
  id: string;
  name: string;
  species: string;
  breed: string;
  lastCheckup: string;
  skinCondition: string;
  status: 'healthy' | 'needs-attention' | 'treatment';
  image: string;
}

export default function Dashboard() {
  const stats = [
    { title: "Consultations", value: "12", icon: Calendar, color: "from-blue-500 to-teal-500" },
    { title: "Products Used", value: "8", icon: Activity, color: "from-purple-500 to-indigo-500" },
    { title: "Notifications", value: "5", icon: Bell, color: "from-amber-500 to-orange-500" }
  ];

  const recentActivities = [
    { title: "AI Consultation", time: "2 hours ago", type: "consultation" },
    { title: "Purchased New Products", time: "1 day ago", type: "purchase" },
    { title: "Expert Consultation", time: "2 days ago", type: "consultation" }
  ];

  const petProfiles: PetProfile[] = [
    {
      id: "1",
      name: "Max",
      species: "Dog",
      breed: "Golden Retriever",
      lastCheckup: "2024-03-15",
      skinCondition: "Mild Allergies",
      status: "needs-attention",
      image: "https://images.unsplash.com/photo-1633722715463-d30f4f325e24?w=400"
    },
    {
      id: "2",
      name: "Luna",
      species: "Cat",
      breed: "Persian",
      lastCheckup: "2024-03-10",
      skinCondition: "Healthy",
      status: "healthy",
      image: "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400"
    },
    {
      id: "3",
      name: "Rocky",
      species: "Dog",
      breed: "Labrador",
      lastCheckup: "2024-03-01",
      skinCondition: "Under Treatment",
      status: "treatment",
      image: "https://images.unsplash.com/photo-1552053831-71594a27632d?w=400"
    }
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="mb-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-4 mb-6"
        >
          <div className="w-16 h-16 bg-gradient-to-r from-pink-500 to-rose-500 rounded-full flex items-center justify-center">
            <User className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Welcome back, User!</h1>
            <p className="text-gray-600">Here's an overview of your pet's skin health journey</p>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
        >
          {stats.map((stat, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className="bg-white rounded-2xl shadow-lg p-6"
            >
              <div className="flex items-center justify-between mb-4">
                <div className={`w-12 h-12 bg-gradient-to-r ${stat.color} rounded-xl flex items-center justify-center`}>
                  <stat.icon className="w-6 h-6 text-white" />
                </div>
                <span className="text-3xl font-bold text-gray-900">{stat.value}</span>
              </div>
              <h3 className="text-gray-600">{stat.title}</h3>
            </motion.div>
          ))}
        </motion.div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="bg-white rounded-2xl shadow-lg p-6"
        >
          <h2 className="text-xl font-semibold mb-4">Recent Activity</h2>
          <div className="space-y-4">
            {recentActivities.map((activity, index) => (
              <div key={index} className="flex items-center justify-between p-4 bg-gray-50 rounded-xl">
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 rounded-full bg-teal-500" />
                  <span className="font-medium">{activity.title}</span>
                </div>
                <span className="text-sm text-gray-500">{activity.time}</span>
              </div>
            ))}
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="bg-white rounded-2xl shadow-lg p-6"
        >
          <h2 className="text-xl font-semibold mb-4">Your Pet's Skin Wellness Journey</h2>
          <div className="space-y-4">
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-600">Treatment Consistency</span>
                <span className="font-medium">100%</span>
              </div>
              <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                <div className="h-full w-[85%] bg-gradient-to-r from-teal-500 to-blue-500 rounded-full" />
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-600">Medication & Product Usage</span>
                <span className="font-medium">92%</span>
              </div>
              <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                <div className="h-full w-[92%] bg-gradient-to-r from-pink-500 to-rose-500 rounded-full" />
              </div>
            </div>
          </div>
        </motion.div>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
        className="mt-8 bg-white rounded-2xl shadow-lg p-6"
      >
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-xl font-semibold">Pet Profiles</h2>
          <button className="px-4 py-2 bg-pink-600 text-white rounded-lg hover:bg-pink-700 transition-colors text-sm">
            Add New Pet
          </button>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {petProfiles.map((pet) => (
            <div key={pet.id} className="bg-gray-50 rounded-xl p-4">
              <div className="flex items-center space-x-4">
                <img 
                  src={pet.image} 
                  alt={pet.name}
                  className="w-16 h-16 rounded-full object-cover"
                />
                <div>
                  <h3 className="font-semibold text-lg">{pet.name}</h3>
                  <p className="text-gray-600 text-sm">{pet.species} - {pet.breed}</p>
                </div>
              </div>
              <div className="mt-4 space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Last Checkup:</span>
                  <span>{new Date(pet.lastCheckup).toLocaleDateString()}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Skin Condition:</span>
                  <span>{pet.skinCondition}</span>
                </div>
                <div className="pt-2">
                  <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                    pet.status === 'healthy' ? 'bg-green-100 text-green-800' :
                    pet.status === 'needs-attention' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-pink-100 text-pink-800'
                  }`}>
                    {pet.status.split('-').map(word => 
                      word.charAt(0).toUpperCase() + word.slice(1)
                    ).join(' ')}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  );
}