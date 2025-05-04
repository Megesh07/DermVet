import React from 'react';
import { User, Bell, Shield, Palette, HelpCircle, LogOut } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';

interface SettingOption {
  icon: React.ReactNode;
  title: string;
  description: string;
  onClick: () => void;
}

export default function SettingsPage() {
  const { logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/auth');
  };

  const settingOptions: SettingOption[] = [
    {
      icon: <User className="w-6 h-6 text-[#D2691E]" />,
      title: "Account Settings",
      description: "Manage your profile and preferences",
      onClick: () => console.log("Account settings clicked")
    },
    {
      icon: <Bell className="w-6 h-6 text-[#D2691E]" />,
      title: "Notifications",
      description: "Configure your notification preferences",
      onClick: () => console.log("Notifications clicked")
    },
    {
      icon: <Shield className="w-6 h-6 text-[#D2691E]" />,
      title: "Privacy & Security",
      description: "Manage your security settings and privacy preferences",
      onClick: () => console.log("Privacy clicked")
    },
    {
      icon: <Palette className="w-6 h-6 text-[#D2691E]" />,
      title: "Appearance",
      description: "Customize your app appearance",
      onClick: () => console.log("Appearance clicked")
    },
    {
      icon: <HelpCircle className="w-6 h-6 text-[#D2691E]" />,
      title: "Help & Support",
      description: "Get help and contact support",
      onClick: () => console.log("Help clicked")
    }
  ];

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h1 className="text-2xl font-bold bg-gradient-to-r from-pink-500 to-[#D2691E] bg-clip-text text-transparent mb-6">
          Settings
        </h1>

        <div className="space-y-4">
          {settingOptions.map((option, index) => (
            <button
              key={index}
              onClick={option.onClick}
              className="w-full p-4 bg-[#FFF5EE] rounded-lg hover:shadow-md transition-all duration-300 flex items-start gap-4 group"
            >
              <div className="mt-1">{option.icon}</div>
              <div className="text-left flex-1">
                <h3 className="text-lg font-semibold text-[#8B4513] group-hover:text-[#D2691E] transition-colors">
                  {option.title}
                </h3>
                <p className="text-gray-600 text-sm">
                  {option.description}
                </p>
              </div>
            </button>
          ))}

          <button
            onClick={handleLogout}
            className="w-full mt-6 p-4 border-2 border-pink-200 rounded-lg hover:border-pink-500 hover:shadow-md transition-all duration-300 flex items-center gap-4 group"
          >
            <LogOut className="w-6 h-6 text-pink-500" />
            <div className="text-left">
              <h3 className="text-lg font-semibold text-pink-500">
                Log Out
              </h3>
              <p className="text-gray-600 text-sm">
                Sign out of your account
              </p>
            </div>
          </button>
        </div>
      </div>
    </div>
  );
} 