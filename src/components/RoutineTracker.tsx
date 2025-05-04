import React, { useState, useEffect } from 'react';
import { Sun, Moon, Check, Plus, Trash2, AlertCircle } from 'lucide-react';
import { format } from 'date-fns';
import { motion, AnimatePresence } from 'framer-motion';

interface RoutineStep {
  id: string;
  name: string;
  time: 'morning' | 'night';
  completed: boolean;
  timestamp?: string;
}

interface RoutineHistory {
  date: string;
  completedSteps: number;
  totalSteps: number;
}

export default function RoutineTracker() {
  const [routineSteps, setRoutineSteps] = useState<RoutineStep[]>(() => {
    const saved = localStorage.getItem('routineSteps');
    return saved ? JSON.parse(saved) : [
      { id: '1', name: 'Cleanse', time: 'morning', completed: false },
      { id: '2', name: 'Tone', time: 'morning', completed: false },
      { id: '3', name: 'Moisturize', time: 'morning', completed: false },
      { id: '4', name: 'Sunscreen', time: 'morning', completed: false },
      { id: '5', name: 'Cleanse', time: 'night', completed: false },
      { id: '6', name: 'Serum', time: 'night', completed: false },
      { id: '7', name: 'Night Cream', time: 'night', completed: false }
    ];
  });

  const [history, setHistory] = useState<RoutineHistory[]>(() => {
    const saved = localStorage.getItem('routineHistory');
    return saved ? JSON.parse(saved) : [];
  });

  const [newStep, setNewStep] = useState('');
  const [selectedTime, setSelectedTime] = useState<'morning' | 'night'>('morning');
  const [showAlert, setShowAlert] = useState(false);
  const [alertMessage, setAlertMessage] = useState('');

  useEffect(() => {
    localStorage.setItem('routineSteps', JSON.stringify(routineSteps));
  }, [routineSteps]);

  useEffect(() => {
    localStorage.setItem('routineHistory', JSON.stringify(history));
  }, [history]);

  const toggleComplete = (id: string) => {
    const now = new Date().toISOString();
    const updatedSteps = routineSteps.map(step =>
      step.id === id
        ? { ...step, completed: !step.completed, timestamp: !step.completed ? now : undefined }
        : step
    );
    setRoutineSteps(updatedSteps);

    // Update history with the new completion status
    const today = format(new Date(), 'yyyy-MM-dd');
    const completedCount = updatedSteps.filter(step => step.completed).length;
    const totalSteps = updatedSteps.length;

    setHistory(prev => {
      const existingEntry = prev.find(h => h.date === today);
      if (existingEntry) {
        return prev.map(h =>
          h.date === today
            ? { ...h, completedSteps: completedCount, totalSteps }
            : h
        );
      }
      return [...prev, { date: today, completedSteps: completedCount, totalSteps }];
    });
  };

  const addStep = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newStep.trim()) {
      showAlertMessage('Please enter a step name');
      return;
    }

    const existingStep = routineSteps.find(
      step => step.name.toLowerCase() === newStep.toLowerCase() && step.time === selectedTime
    );

    if (existingStep) {
      showAlertMessage(`${newStep} already exists in your ${selectedTime} routine`);
      return;
    }

    const newId = String(Date.now());
    setRoutineSteps(steps => [...steps, {
      id: newId,
      name: newStep,
      time: selectedTime,
      completed: false
    }]);
    setNewStep('');
  };

  const deleteStep = (id: string) => {
    setRoutineSteps(steps => steps.filter(step => step.id !== id));
  };

  const showAlertMessage = (message: string) => {
    setAlertMessage(message);
    setShowAlert(true);
    setTimeout(() => setShowAlert(false), 3000);
  };

  const calculateProgress = (time: 'morning' | 'night') => {
    const timeSteps = routineSteps.filter(step => step.time === time);
    if (timeSteps.length === 0) return 0;
    const completed = timeSteps.filter(step => step.completed).length;
    return (completed / timeSteps.length) * 100;
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <AnimatePresence>
        {showAlert && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="fixed top-4 right-4 bg-pink-100 border border-pink-400 text-pink-700 px-4 py-3 rounded-lg flex items-center z-50"
          >
            <AlertCircle className="w-5 h-5 mr-2" />
            {alertMessage}
          </motion.div>
        )}
      </AnimatePresence>

      <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-pink-500 to-[#D2691E] bg-clip-text text-transparent">
            Skincare Routine Tracker
          </h1>
          <span className="text-lg text-[#8B4513]">{format(new Date(), 'MMMM d, yyyy')}</span>
        </div>

        <form onSubmit={addStep} className="mb-8">
          <div className="flex gap-4 flex-wrap md:flex-nowrap">
            <input
              type="text"
              value={newStep}
              onChange={(e) => setNewStep(e.target.value)}
              placeholder="Add new routine step..."
              className="flex-1 p-3 border border-pink-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-pink-500 min-w-[200px] bg-[#FFF5EE]"
            />
            <select
              value={selectedTime}
              onChange={(e) => setSelectedTime(e.target.value as 'morning' | 'night')}
              className="p-3 border border-pink-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-pink-500 bg-[#FFF5EE]"
            >
              <option value="morning">Morning</option>
              <option value="night">Night</option>
            </select>
            <button
              type="submit"
              className="bg-gradient-to-r from-pink-500 to-[#D2691E] text-white px-6 py-3 rounded-lg hover:shadow-lg transition-all duration-300 flex items-center whitespace-nowrap"
            >
              <Plus className="w-5 h-5 mr-2" />
              Add Step
            </button>
          </div>
        </form>

        <div className="grid md:grid-cols-2 gap-8">
          {['morning', 'night'].map((time) => (
            <div key={time} className="bg-gradient-to-br from-[#FFF5EE] to-white rounded-xl p-6 shadow-sm">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center">
                  {time === 'morning' ? (
                    <Sun className="w-6 h-6 text-[#D2691E] mr-2" />
                  ) : (
                    <Moon className="w-6 h-6 text-[#D2691E] mr-2" />
                  )}
                  <h2 className="text-xl font-semibold text-[#8B4513] capitalize">
                    {time} Routine
                  </h2>
                </div>
                <span className="text-sm font-medium px-3 py-1 bg-white rounded-full text-[#8B4513] shadow-sm">
                  {Math.round(calculateProgress(time as 'morning' | 'night'))}% Complete
                </span>
              </div>

              <div className="h-2 bg-pink-100 rounded-full mb-4 overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${calculateProgress(time as 'morning' | 'night')}%` }}
                  transition={{ duration: 0.5 }}
                  className="h-full bg-gradient-to-r from-pink-500 to-[#D2691E] rounded-full"
                />
              </div>

              <div className="space-y-3">
                {routineSteps
                  .filter(step => step.time === time)
                  .map(step => (
                    <motion.div
                      key={step.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      className="flex items-center p-4 bg-white rounded-lg group hover:shadow-md transition-all duration-300"
                    >
                      <button
                        onClick={() => toggleComplete(step.id)}
                        className={`w-6 h-6 rounded-full mr-3 flex items-center justify-center transition-all duration-300
                          ${step.completed 
                            ? 'bg-gradient-to-r from-pink-500 to-[#D2691E] text-white scale-110' 
                            : 'border-2 border-pink-300 hover:border-pink-500'}`}
                      >
                        {step.completed && <Check className="w-4 h-4" />}
                      </button>
                      <span className={`flex-1 transition-all duration-300 ${
                        step.completed ? 'line-through text-gray-400' : 'text-[#8B4513]'
                      }`}>
                        {step.name}
                      </span>
                      {step.timestamp && (
                        <span className="text-xs bg-pink-50 text-pink-600 px-2 py-1 rounded-full mr-2">
                          {format(new Date(step.timestamp), 'h:mm a')}
                        </span>
                      )}
                      <button
                        onClick={() => deleteStep(step.id)}
                        className="opacity-0 group-hover:opacity-100 text-pink-500 hover:text-pink-700 transition-all duration-300 p-1 hover:bg-pink-50 rounded-full"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </motion.div>
                  ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {history.length > 0 && (
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-6 text-[#8B4513]">Progress History</h2>
          <div className="space-y-4">
            {history.slice(-7).reverse().map((entry) => (
              <div key={entry.date} className="flex items-center">
                <span className="w-24 text-sm font-medium text-[#8B4513]">
                  {format(new Date(entry.date), 'MMM d')}
                </span>
                <div className="flex-1 h-2 bg-pink-100 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${(entry.completedSteps / entry.totalSteps) * 100}%` }}
                    transition={{ duration: 0.5 }}
                    className="h-full bg-gradient-to-r from-pink-500 to-[#D2691E] rounded-full"
                  />
                </div>
                <span className="w-16 text-right text-sm font-medium text-[#8B4513]">
                  {Math.round((entry.completedSteps / entry.totalSteps) * 100)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}