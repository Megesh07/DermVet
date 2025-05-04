import React, { useState } from 'react';
import { motion } from 'framer-motion';

interface ClientRequest {
  id: string;
  clientName: string;
  concern: string;
  description: string;
  images?: string[];
  status: 'pending' | 'accepted' | 'completed';
  bidAmount?: number;
}

export default function ClientRequests() {
  const [requests, setRequests] = useState<ClientRequest[]>([
    // Sample data - replace with actual API calls
    {
      id: '1',
      clientName: 'John Doe',
      concern: 'Acne Treatment',
      description: 'Experiencing severe acne breakouts...',
      status: 'pending'
    },
    // Add more sample requests
  ]);

  const [selectedRequest, setSelectedRequest] = useState<string | null>(null);
  const [bidAmount, setBidAmount] = useState<number>(0);

  const handleBid = (requestId: string) => {
    if (bidAmount <= 0) return;

    setRequests(prev => prev.map(req => 
      req.id === requestId 
        ? { ...req, status: 'accepted', bidAmount } 
        : req
    ));
    setSelectedRequest(null);
    setBidAmount(0);
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold text-gray-900 mb-6">Client Requests</h1>
      
      <div className="space-y-4">
        {requests.map(request => (
          <motion.div
            key={request.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-lg shadow-md p-6"
          >
            <div className="flex justify-between items-start">
              <div>
                <h2 className="text-lg font-semibold text-gray-900">{request.concern}</h2>
                <p className="text-gray-600 mt-1">{request.description}</p>
                <p className="text-sm text-gray-500 mt-2">Client: {request.clientName}</p>
              </div>
              <span className={`px-3 py-1 rounded-full text-sm ${
                request.status === 'pending' ? 'bg-yellow-100 text-yellow-800' :
                request.status === 'accepted' ? 'bg-green-100 text-green-800' :
                'bg-blue-100 text-blue-800'
              }`}>
                {request.status}
              </span>
            </div>

            {request.status === 'pending' && (
              <div className="mt-4">
                {selectedRequest === request.id ? (
                  <div className="flex items-center space-x-4">
                    <input
                      type="number"
                      min="0"
                      value={bidAmount}
                      onChange={(e) => setBidAmount(Number(e.target.value))}
                      className="border rounded px-3 py-1 w-32"
                      placeholder="Bid amount"
                    />
                    <button
                      onClick={() => handleBid(request.id)}
                      className="bg-pink-600 text-white px-4 py-1 rounded hover:bg-pink-700"
                    >
                      Submit Bid
                    </button>
                    <button
                      onClick={() => setSelectedRequest(null)}
                      className="text-gray-600 hover:text-gray-800"
                    >
                      Cancel
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={() => setSelectedRequest(request.id)}
                    className="text-pink-600 hover:text-pink-700"
                  >
                    Place Bid
                  </button>
                )}
              </div>
            )}

            {request.bidAmount && (
              <div className="mt-2 text-sm text-gray-600">
                Your bid: ${request.bidAmount}
              </div>
            )}
          </motion.div>
        ))}
      </div>
    </div>
  );
} 