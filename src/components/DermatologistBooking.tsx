import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  User, Phone, Mail, MessageSquare, MapPin, Star, 
  Clock, Calendar, Stethoscope, MessageCircle, 
  Video, Check, X, AlertCircle
} from 'lucide-react';

// Add this interface at the top of the file, after the imports
interface Doctor {
  id: string;
  name: string;
  qualification: string;
  specialization: string;
  experience: number;
  location: string;
  consultationFee: number;
  rating: number;
  reviews: number;
  image: string;
  isOnline: boolean;
  expertise: string[];
}

const mockDoctors: Doctor[] = [
    {
      id: '1',
      name: 'Dr. Diu Somani',
      qualification: 'BVSc & AH, MVSc (Dermatology)',
      specialization: 'Veterinary Dermatology',
      experience: 8,
      location: 'Mumbai, India',
      consultationFee: 5000,
      rating: 4.8,
      reviews: 120,
      image: 'https://www.felican.in/img/sunil2.jpg', 
      isOnline: true,
      expertise: ['Canine Dermatitis', 'Feline Skin Disorders', 'Allergic Skin Diseases']
    },
    {
      id: '2',
      name: 'Dr. Ravi Murarka',
      qualification: 'BVSc & AH, MVSc (Dermatology)',
      specialization: 'Veterinary Dermatology',
      experience: 15,
      location: 'Delhi, India',
      consultationFee: 5500,
      rating: 4.9,
      reviews: 150,
      image: 'https://th.bing.com/th/id/OIP.7B0i61cNBj8CGvInfw_eFAHaE7?rs=1&pid=ImgDetMain', 
      isOnline: false,
      expertise: ['Parasitic Skin Infections', 'Fungal Skin Diseases', 'Bacterial Dermatitis']
    },
    {
      id: '3',
      name: 'Dr. Sharmila Nayak',
      qualification: 'BVSc & AH, MVSc (Dermatology)',
      specialization: 'Veterinary Dermatology',
      experience: 10,
      location: 'Bangalore, India',
      consultationFee: 4800,
      rating: 4.7,
      reviews: 110,
      image: 'https://img.freepik.com/premium-photo/young-indian-veterinarian-woman-doctor-with-dog-inside-clinic-with-happy-expression_466689-95909.jpg', 
      isOnline: true,
      expertise: ['Skin Allergies', 'Dermatopathology', 'Autoimmune Skin Diseases']
    },
    {
      id: '4',
      name: 'Dr. Chandana Rao',
      qualification: 'BVSc & AH, MVSc (Dermatology)',
      specialization: 'Veterinary Dermatology',
      experience: 12,
      location: 'Hyderabad, India',
      consultationFee: 5300,
      rating: 4.8,
      reviews: 130,
      image: 'https://uploads.sarvgyan.com/2016/03/Veterinary.jpg', 
      isOnline: false,
      expertise: ['Mange', 'Ringworm', 'Tick-borne Skin Issues']
    },
    {
      id: '5',
      name: 'Dr. Vishal Iyer',
      qualification: 'BVSc & AH, MVSc (Dermatology)',
      specialization: 'Companion Animal Dermatology',
      experience: 9,
      location: 'Chennai, India',
      consultationFee: 4700,
      rating: 4.6,
      reviews: 100,
      image: 'https://dccwebsiteimages.s3.ap-south-1.amazonaws.com/585_585_7ccac79c3b.png', 
      isOnline: true,
      expertise: ['Psoriasis', 'Vitiligo', 'Skin Allergies']
    },
    {
      id: '6',
      name: 'Dr. Meera Kapoor',
      qualification: 'BVSc & AH, MVSc (Dermatology)',
      specialization: 'Exotic Pet Dermatology',
      experience: 14,
      location: 'Pune, India',
      consultationFee: 6000,
      rating: 4.9,
      reviews: 140,
      image: 'https://aavio.org/wp-content/uploads/2020/08/Sudha-Komma-300x274.jpeg', 
      isOnline: false,
      expertise: ['Reptile Skin Diseases', 'Feather Loss in Birds', 'Scaly Skin in Turtles']
    },
    {
      id: '7',
      name: 'Dr. Karthik Srinivasan',
      qualification: 'BVSc & AH, MVSc (Dermatology)',
      specialization: 'Veterinary Allergy & Immunology',
      experience: 13,
      location: 'Kolkata, India',
      consultationFee: 5100,
      rating: 4.7,
      reviews: 115,
      image: 'https://th.bing.com/th/id/R.41eb163b6b330a7c51d95562715b66e6?rik=noZ9Ai0kajwWLQ&riu=http%3a%2f%2fpetbizindia.com%2fwp-content%2fuploads%2f2024%2f02%2fDr-Vinod-Sharma-DCC-HOspital.jpg&ehk=dISek4l%2bFDI8603eNazDRJw1gGy2oy2LhfVPuA2xxXc%3d&risl=&pid=ImgRaw&r=0',
      isOnline: true,
      expertise: ['Flea Allergies', 'Food Allergies', 'Atopic Dermatitis']
    },
    {
      id: '8',
      name: 'Dr. Ritu Sharma',
      qualification: 'BVSc & AH, MVSc (Dermatology)',
      specialization: 'Bacterial & Fungal Skin Infections',
      experience: 11,
      location: 'Ahmedabad, India',
      consultationFee: 5200,
      rating: 4.8,
      reviews: 125,
      image: 'https://th.bing.com/th/id/R.fc381aaa4f470a1d6554b31021a97598?rik=5d45ON5GVN6n9g&riu=http%3a%2f%2fim.rediff.com%2fgetahead%2f2013%2fjul%2f12vet3.jpg&ehk=NelROs7HoLk%2b1xXo0PpxRixssoIHe8TFBAuxru7tYos%3d&risl=&pid=ImgRaw&r=0&sres=1&sresct=1',
      isOnline: false,
      expertise: ['Hot Spots', 'Pyoderma', 'Yeast Infections']
    },
    {
      id: '9',
      name: 'Dr. Arjun Nair',
      qualification: 'BVSc & AH, MVSc (Dermatology)',
      specialization: 'Veterinary Dermatologic Surgery',
      experience: 16,
      location: 'Jaipur, India',
      consultationFee: 5800,
      rating: 4.9,
      reviews: 135,
      image: 'https://images.indianexpress.com/2022/10/pet-jaguar-and-panther.jpg',
      isOnline: true,
      expertise: ['Skin Tumor Removal', 'Mast Cell Tumors', 'Surgical Skin Repair']
    },
    {
      id: '10',
      name: 'Dr. Neha Bansal',
      qualification: 'BVSc & AH, MVSc (Dermatology)',
      specialization: 'Canine & Feline Dermatology',
      experience: 12,
      location: 'Lucknow, India',
      consultationFee: 4900,
      rating: 4.6,
      reviews: 105,
      image: 'https://i.ytimg.com/vi/zDmQ7HSG_bk/maxresdefault.jpg',
      isOnline: true,
      expertise: ['Ear Infections', 'Itchy Skin in Dogs', 'Seborrhea']
    },
    {
      id: '11',
      name: 'Dr. Ananya Mehta',
      qualification: 'BVSc & AH, MVSc (Dermatology)',
      specialization: 'Small Animal Dermatology',
      experience: 7,
      location: 'Indore, India',
      consultationFee: 4800,
      rating: 4.7,
      reviews: 112,
      image: 'https://api.dccpets.in/uploads/userprofiles/21087839741649319434.jpeg',
      isOnline: true,
      expertise: ['Skin Infections', 'Canine Atopic Dermatitis', 'Fungal Skin Diseases']
    },
    {
      id: '12',
      name: 'Dr. Rajesh Verma',
      qualification: 'BVSc & AH, MVSc (Dermatology)',
      specialization: 'Companion Animal Dermatology',
      experience: 14,
      location: 'Nagpur, India',
      consultationFee: 5500,
      rating: 4.9,
      reviews: 155,
      image: 'https://media.istockphoto.com/photos/veterinary-doctor-does-medical-examination-on-a-yellow-labrador-picture-id485428168?b=1&k=20&m=485428168&s=170667a&w=0&h=aUjNwdLt5LkUlVczbgU9qWcxHY1WXuPav3N4P6AlFjo=',
      isOnline: false,
      expertise: ['Parasitic Infections', 'Food Allergies', 'Mange Treatment']
    },
    {
      id: '13',
      name: 'Dr. Sneha Patil',
      qualification: 'BVSc & AH, MVSc (Dermatology)',
      specialization: 'Veterinary Dermatology & Surgery',
      experience: 10,
      location: 'Thane, India',
      consultationFee: 5300,
      rating: 4.8,
      reviews: 138,
      image: 'https://th.bing.com/th/id/OIP.I5kvfbn7q6GPm-nvu0XhwQHaGJ?w=603&h=500&rs=1&pid=ImgDetMain',
      isOnline: true,
      expertise: ['Skin Tumors', 'Surgical Dermatology', 'Autoimmune Skin Disorders']
    },
    {
      id: '14',
      name: 'Dr. Manoj Reddy',
      qualification: 'BVSc & AH, MVSc (Dermatology)',
      specialization: 'Exotic Pet Dermatology',
      experience: 9,
      location: 'Vijayawada, India',
      consultationFee: 5000,
      rating: 4.7,
      reviews: 121,
      image: 'https://im.indiatimes.in/content/2022/Oct/310230850_493897316068560_2102750959409685871_n_633d6cc87cf80.jpg',
      isOnline: false,
      expertise: ['Feather Plucking in Birds', 'Reptile Skin Issues', 'Scale Rot']
    },
    {
      id: '15',
      name: 'Dr. Pooja Sharma',
      qualification: 'BVSc & AH, MVSc (Dermatology)',
      specialization: 'Veterinary Allergy & Immunology',
      experience: 11,
      location: 'Surat, India',
      consultationFee: 5200,
      rating: 4.8,
      reviews: 130,
      image: 'https://th.bing.com/th/id/OIP.qvlwv1RTt9XOm8V_3ArpsQHaIl?w=750&h=870&rs=1&pid=ImgDetMain',
      isOnline: true,
      expertise: ['Pet Skin Allergies', 'Food Intolerance', 'Flea Dermatitis']
    },
    {
      id: '16',
      name: 'Dr. Arvind Kumar',
      qualification: 'BVSc & AH, MVSc (Dermatology)',
      specialization: 'Bacterial & Fungal Skin Infections',
      experience: 15,
      location: 'Patna, India',
      consultationFee: 5800,
      rating: 4.9,
      reviews: 145,
      image: 'https://avcorlando.com/static/e91c890c601e119e574ba6279caaf9a8/d75e0/m2.jpg',
      isOnline: false,
      expertise: ['Hot Spots', 'Pyoderma', 'Yeast Infections']
    },
    {
      id: '17',
      name: 'Dr. Priya Menon',
      qualification: 'BVSc & AH, MVSc (Dermatology)',
      specialization: 'Canine & Feline Dermatology',
      experience: 8,
      location: 'Bhopal, India',
      consultationFee: 4900,
      rating: 4.6,
      reviews: 109,
      image: 'https://aavio.org/wp-content/uploads/2020/08/Sudha-Komma.jpeg',
      isOnline: true,
      expertise: ['Ear Infections', 'Seborrhea', 'Itchy Skin in Dogs']
    },
    {
      id: '18',
      name: 'Dr. Kunal Desai',
      qualification: 'BVSc & AH, MVSc (Dermatology)',
      specialization: 'Veterinary Dermatopathology',
      experience: 13,
      location: 'Nashik, India',
      consultationFee: 5100,
      rating: 4.7,
      reviews: 119,
      image: 'https://static.wixstatic.com/media/ba2cd3_b001d1e8c1704ebba337da6d5fdbaef5~mv2.jpg/v1/fill/w_370,h_463,al_c,q_80,enc_auto/ba2cd3_b001d1e8c1704ebba337da6d5fdbaef5~mv2.jpg',
      isOnline: false,
      expertise: ['Autoimmune Skin Diseases', 'Dermatologic Cytology', 'Skin Tumor Biopsies']
    },
    {
      id: '19',
      name: 'Dr. Nidhi Kapoor',
      qualification: 'BVSc & AH, MVSc (Dermatology)',
      specialization: 'Veterinary Dermatology & Surgery',
      experience: 12,
      location: 'Goa, India',
      consultationFee: 5300,
      rating: 4.8,
      reviews: 127,
      image: 'https://th.bing.com/th/id/OIP.xHKiJH2NiDZiCk9bfgT_6gHaHa?w=600&h=600&rs=1&pid=ImgDetMain',
      isOnline: true,
      expertise: ['Skin Grafts', 'Mast Cell Tumors', 'Wound Healing in Pets']
    },
    {
      id: '20',
      name: 'Dr. Sandeep Joshi',
      qualification: 'BVSc & AH, MVSc (Dermatology)',
      specialization: 'Small Animal Dermatology',
      experience: 9,
      location: 'Coimbatore, India',
      consultationFee: 4900,
      rating: 4.6,
      reviews: 105,
      image: 'https://th.bing.com/th/id/OIP.DTc6IAywSfOgx32GNBz1YwHaHa?w=500&h=500&rs=1&pid=ImgDetMain',
      isOnline: false,
      expertise: ['Dandruff in Pets', 'Itchy Skin Treatment', 'Skin Hydration for Pets']
    }
 
];

export default function DermatologistBooking() {
  const [selectedDoctor, setSelectedDoctor] = useState<Doctor | null>(null);
  const [showAppointmentModal, setShowAppointmentModal] = useState(false);
  const [skinIssue, setSkinIssue] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredDoctors, setFilteredDoctors] = useState(mockDoctors);
  const [showSuccessModal, setShowSuccessModal] = useState(false);

  useEffect(() => {
    const filtered = mockDoctors.filter(doctor => 
      doctor.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      doctor.specialization.toLowerCase().includes(searchQuery.toLowerCase()) ||
      doctor.expertise.some(exp => exp.toLowerCase().includes(searchQuery.toLowerCase()))
    );
    setFilteredDoctors(filtered);
  }, [searchQuery]);

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Search Bar */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-6">Find a Veterinary Dermatologist</h1>
        <input
          type="text"
          placeholder="Search by name, specialization, or skin condition..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-pink-300"
        />
      </div>

      {/* Doctor List */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredDoctors.map((doctor) => (
          <motion.div
            key={doctor.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-xl shadow-lg overflow-hidden"
          >
            {/* Keep existing doctor card content */}
            <div className="relative">
              <img
                src={doctor.image}
                alt={doctor.name}
                className="w-full h-48 object-cover"
              />
              {doctor.isOnline && (
                <span className="absolute top-4 right-4 px-3 py-1 bg-green-500 text-white text-sm rounded-full flex items-center">
                  <span className="w-2 h-2 bg-white rounded-full mr-2" />
                  Online
                </span>
              )}
            </div>
            
            <div className="p-6">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h3 className="text-xl font-semibold">{doctor.name}</h3>
                  <p className="text-gray-600">{doctor.qualification}</p>
                </div>
                <div className="flex items-center">
                  <Star className="w-5 h-5 text-yellow-400 mr-1" />
                  <span className="font-semibold">{doctor.rating}</span>
                  <span className="text-gray-500 text-sm ml-1">({doctor.reviews})</span>
                </div>
              </div>

              <div className="space-y-2 mb-4">
                <div className="flex items-center text-gray-600">
                  <MapPin className="w-5 h-5 mr-2" />
                  {doctor.location}
                </div>
                <div className="flex items-center text-gray-600">
                  <Clock className="w-5 h-5 mr-2" />
                  {doctor.experience} years experience
                </div>
                <div className="flex items-center text-gray-600">
                  <MessageSquare className="w-5 h-5 mr-2" />
                  {doctor.specialization}
                </div>
              </div>

              <div className="mb-4">
                <h4 className="font-medium mb-2">Expertise</h4>
                <div className="flex flex-wrap gap-2">
                  {doctor.expertise.map((exp, index) => (
                    <span
                      key={index}
                      className="px-3 py-1 bg-pink-100 text-pink-700 rounded-full text-sm"
                    >
                      {exp}
                    </span>
                  ))}
                </div>
              </div>

              <div className="flex items-center justify-between mt-4">
                <span className="text-2xl font-bold text-pink-600">
                  ₹{doctor.consultationFee}
                </span>
                <button
                  onClick={() => {
                    setSelectedDoctor(doctor);
                    setShowAppointmentModal(true);
                  }}
                  className="px-6 py-2 bg-pink-600 text-white rounded-lg hover:bg-pink-700 transition-colors"
                >
                  Book Now
                </button>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Keep existing Appointment Modal */}
      <AnimatePresence>
        {showAppointmentModal && selectedDoctor && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50"
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-white rounded-xl p-6 max-w-lg w-full"
            >
              <div className="flex justify-between items-start mb-6">
                <h2 className="text-2xl font-semibold">Book Appointment</h2>
                <button
                  onClick={() => setShowAppointmentModal(false)}
                  className="text-gray-500 hover:text-gray-700"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Describe your skin issue
                  </label>
                  <textarea
                    value={skinIssue}
                    onChange={(e) => setSkinIssue(e.target.value)}
                    className="w-full p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-pink-300 h-32"
                    placeholder="Please provide details about your skin concern..."
                    required
                  />
                </div>

                <div className="flex gap-4">
                  <button
                    onClick={() => {
                      setShowAppointmentModal(false);
                      setShowSuccessModal(true);
                      setSkinIssue('');
                    }}
                    className="flex-1 px-6 py-3 bg-pink-600 text-white rounded-lg hover:bg-pink-700 transition-colors"
                  >
                    Confirm Booking
                  </button>
                  <button
                    onClick={() => setShowAppointmentModal(false)}
                    className="px-6 py-3 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Keep existing Success Modal */}
      <AnimatePresence>
        {showSuccessModal && selectedDoctor && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50"
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-white rounded-xl p-6 max-w-md w-full text-center"
            >
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Check className="w-8 h-8 text-green-500" />
              </div>
              <h2 className="text-2xl font-semibold mb-2">Appointment Booked!</h2>
              <p className="text-gray-600 mb-6">
                {selectedDoctor.name} will contact you soon to confirm your appointment.
              </p>
              <button
                onClick={() => {
                  setShowSuccessModal(false);
                  setSelectedDoctor(null);
                }}
                className="px-6 py-3 bg-pink-600 text-white rounded-lg hover:bg-pink-700 transition-colors"
              >
                Done
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}