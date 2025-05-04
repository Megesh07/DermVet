import { useState } from 'react';
import { ShoppingCart } from 'lucide-react';

interface Product {
  id: number;
  name: string;
  type: 'veterinary';
  price: number;
  description: string;
  image: string;
}

const products: Product[] = [
  {
    "id": 1,
    "name": "Flea & Tick Spray",
    "type": "veterinary",
    "price": 899,
    "description": "Fast-acting flea and tick repellent for dogs and cats.",
    "image": "https://th.bing.com/th/id/OIP.qx_e66gGjl3UzEJXh4CfNAHaHa?rs=1&pid=ImgDetMain"
  },
  {
    "id": 2,
    "name": "Herbal Wound Healing Balm",
    "type": "veterinary",
    "price": 599,
    "description": "Natural antiseptic balm with neem and turmeric for wound healing.",
    "image": "https://sammcgeesalaska.com/wp-content/uploads/2017/11/DSC_0464-e14914491724691-scaled.jpg"
  },
  {
    "id": 3,
    "name": "Joint Support Tablets",
    "type": "veterinary",
    "price": 1299,
    "description": "Glucosamine and chondroitin tablets for pet joint health.",
    "image": "https://th.bing.com/th/id/OIP.pP7EKYZQmCTjT8W38hqYVQHaHT?rs=1&pid=ImgDetMain"
  },
  {
    "id": 4,
    "name": "Ayurvedic Digestive Tonic",
    "type": "veterinary",
    "price": 499,
    "description": "Herbal supplement for improving digestion in pets.",
    "image": "https://assets.petco.com/petco/image/upload/f_auto,q_auto/3170339-center-1"
  },
  {
    "id": 5,
    "name": "Antibiotic Skin Ointment",
    "type": "veterinary",
    "price": 799,
    "description": "Effective antibacterial ointment for pet skin infections.",
    "image": "https://th.bing.com/th/id/OIP.8tXXgLnU8VgnwciGp8FSZwHaN0?rs=1&pid=ImgDetMain"
  },
  {
    "id": 6,
    "name": "Neem & Aloe Skin Spray",
    "type": "veterinary",
    "price": 449,
    "description": "Herbal spray for soothing itchy and irritated pet skin.",
    "image": "https://th.bing.com/th/id/OIP.fVdThUWwuSP_89PojjBf2wHaHa?rs=1&pid=ImgDetMain"
  },
  {
    "id": 7,
    "name": "Multivitamin Chewables",
    "type": "veterinary",
    "price": 999,
    "description": "Essential vitamin and mineral supplement for pets.",
    "image": "https://m.media-amazon.com/images/I/71QX9wr-sKS.jpg"
  },
  {
    "id": 8,
    "name": "Turmeric Anti-Inflammatory Powder",
    "type": "veterinary",
    "price": 399,
    "description": "Ayurvedic turmeric powder for reducing pet inflammation.",
    "image": "https://cdn.pipingrock.com/images/product/amazon/product/turmeric-curcumin-powder-organic-16-oz-455-g-bottle-23380.jpg?v=2020112401"
  },
  {
    "id": 9,
    "name": "Neem Skin Soothing Oil",
    "type": "veterinary",
    "price": 349,
    "description": "Neem-based oil to relieve pet itching and skin infections.",
    "image": "https://holistichealthpetcare.com/wp-content/uploads/2021/04/Holistic-Health-Pet-Care-Neem-Oil-768x768.jpg"
  },
  {
    "id": 10,
    "name": "Aloe Vera Hydration Gel",
    "type": "veterinary",
    "price": 299,
    "description": "Aloe vera gel for moisturizing dry and irritated pet skin.",
    "image": " https://th.bing.com/th/id/OIP.x7emP9xSfNrLCv7j2LfAjQAAAA?rs=1&pid=ImgDetMain"
  },
  {
    "id": 11,
    "name": "VetApproved Anti-Fungal Shampoo",
    "type": "veterinary",
    "price": 699,
    "description": "Medicated shampoo for treating fungal and bacterial infections in pets.",
    "image": "https://th.bing.com/th/id/OIP.9ZK6DGhlK9IZd1DvCDZ_XQHaHa?rs=1&pid=ImgDetMain"
  },
  {
    "id": 12,
    "name": "Coconut & Turmeric Balm",
    "type": "veterinary",
    "price": 399,
    "description": "Soothing balm with coconut oil and turmeric for healing skin rashes.",
    "image": "https://www.lovebee.buzz/wp-content/uploads/2017/08/Soothing-Healing-Balm.jpg"
  },
  {
    "id": 13,
    "name": "Herbal Anti-Itch Spray",
    "type": "veterinary",
    "price": 549,
    "description": "Herbal spray to reduce skin allergies and itching in pets.",
    "image": "https://woononapetfoods.com.au/cdn/shop/products/natural-animal-solution-itchy-skin-spray-100ml-221870.jpg?v=1626688668"
  },
  {
    "id": 14,
    "name": "Manuka Honey Wound Cream",
    "type": "veterinary",
    "price": 1499,
    "description": "Natural healing cream with Manuka honey for pet wounds and infections.",
    "image": "https://th.bing.com/th/id/OIP.InDdIGUX9DNGxgtmMEqCYQHaHa?rs=1&pid=ImgDetMain"
  },
  {
    "id": 15,
    "name": "Ayurvedic Pet Skin Detox Tonic",
    "type": "veterinary",
    "price": 649,
    "description": "Oral tonic to detoxify pet skin and improve coat health.",
    "image": "https://th.bing.com/th/id/OIP.GGLvbD1h9HDjRmRCkhXPQgHaHa?rs=1&pid=ImgDetMain"
  },
  {
    "id": 16,
    "name": "VetMed Antibacterial Ointment",
    "type": "veterinary",
    "price": 599,
    "description": "Antibacterial ointment for treating pet hot spots and infections.",
    "image": "https://m.media-amazon.com/images/I/81WYHxUdU8L.jpg"
  },
  {
    "id": 17,
    "name": "Sandalwood & Neem Healing Lotion",
    "type": "veterinary",
    "price": 749,
    "description": "Lotion with sandalwood and neem to soothe irritated pet skin.",
    "image": "https://5.imimg.com/data5/SELLER/Default/2022/9/RJ/XO/AP/7892700/neemli-rose-and-sandalwood-body-lotion-1000x1000.jpg"
  },
  {
    "id": 18,
    "name": "Herbal Flea & Tick Powder",
    "type": "veterinary",
    "price": 449,
    "description": "Herbal powder to repel fleas and ticks naturally in pets.",
    "image": "https://th.bing.com/th/id/OIP.bJCvfs3TbgUtgXOt87OI1AHaJf?rs=1&pid=ImgDetMain"
  }
];

export default function EMart() {
  const [cart, setCart] = useState<Product[]>([]);

  const addToCart = (product: Product) => {
    setCart((prevCart) => [...prevCart, product]);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold bg-gradient-to-r from-pink-500 to-[#D2691E] bg-clip-text text-transparent">
          AnimLuxe
        </h1>
        <div className="flex space-x-4">
          <button className="bg-gradient-to-r from-pink-500 to-[#D2691E] text-white px-4 py-2 rounded-lg flex items-center">
            <ShoppingCart className="w-5 h-5 mr-2" />
            Cart ({cart.length})
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {products.map((product) => (
          <div key={product.id} className="bg-white rounded-lg shadow-lg overflow-hidden flex flex-col">
            <img src={product.image} alt={product.name} className="w-full h-48 object-cover" />
            <div className="p-4 flex-grow flex flex-col justify-between">
              <h3 className="text-lg font-semibold">{product.name}</h3>
              <p className="text-sm text-gray-600">{product.description}</p>
              <span className="text-lg font-bold text-pink-600">₹{product.price}</span>
              <button onClick={() => addToCart(product)} className="mt-4 w-full bg-pink-500 text-white py-2 rounded-lg">
                Add to Cart
              </button>
            </div>
          </div>
        ))}
      </div>

      {cart.length > 0 && (
        <div className="mt-6 text-center">
          <button className="bg-green-500 text-white px-6 py-3 rounded-lg text-lg font-semibold">
            Proceed to Buy ({cart.length} Items)
          </button>
        </div>
      )}
    </div>
  );
}
