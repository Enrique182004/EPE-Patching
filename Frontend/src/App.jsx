import React, { useState, useEffect, useCallback, useRef } from 'react';
import { 
  RefreshCcw, Calendar, CheckCircle, Clock, TrendingDown, AlertTriangle, 
  MessageCircle, Send, X, Minimize2, Maximize2, Zap, Shield, BarChart3,
  Filter, Download, Share2, Sun, Moon
} from 'lucide-react';

// Backend API URL (Flask server runs on port 5001)
const apiBaseUrl = 'https://epe-backend.onrender.com';
// --- Enhanced Confidence Badge with Animation ---
const ConfidenceBadge = ({ confidence }) => {
  let colorClass = 'bg-gray-200 text-gray-800 ring-gray-400';
  let icon = <AlertTriangle size={16} className="text-gray-600 mr-1" />;
  let label = 'Low';

  if (confidence >= 85) {
    colorClass = 'bg-gradient-to-r from-green-400 to-emerald-500 text-white ring-green-500';
    icon = <CheckCircle size={16} className="text-white mr-1" />;
    label = 'High';
  } else if (confidence >= 70) {
    colorClass = 'bg-gradient-to-r from-yellow-400 to-orange-400 text-white ring-yellow-500';
    icon = <Zap size={16} className="text-white mr-1" />;
    label = 'Good';
  } else if (confidence >= 50) {
    colorClass = 'bg-gradient-to-r from-orange-400 to-red-400 text-white ring-orange-500';
    icon = <AlertTriangle size={16} className="text-white mr-1" />;
    label = 'Medium';
  }

  return (
    <div className="flex items-center gap-2">
      <div className={`inline-flex items-center text-sm font-bold px-4 py-1.5 rounded-full ring-2 ring-inset ${colorClass} shadow-md transition-all hover:scale-105`}>
        {icon}
        <span>{confidence}%</span>
      </div>
      <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">{label} Confidence</span>
    </div>
  );
};

// --- Enhanced Prediction Card with Hover Effects ---
const PredictionCard = ({ prediction, index }) => {
  const { date, dayOfWeek, window, confidence, impact } = prediction;
  const [isHovered, setIsHovered] = useState(false);
  
  // Determine if it's a weekend
  const isWeekend = ['Saturday', 'Sunday'].includes(dayOfWeek);
  
  return (
    <div 
      className="group relative bg-white p-6 rounded-2xl shadow-lg transition-all duration-300 hover:shadow-2xl border-2 border-gray-100 hover:border-indigo-300 flex flex-col justify-between h-full cursor-pointer transform hover:-translate-y-1"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      style={{
        animationDelay: `${index * 50}ms`,
        animation: 'slideUp 0.5s ease-out forwards'
      }}
    >
      {/* Rank Badge */}
      {index < 3 && (
        <div className="absolute -top-3 -right-3 bg-gradient-to-br from-indigo-500 to-purple-600 text-white w-10 h-10 rounded-full flex items-center justify-center font-bold text-sm shadow-lg">
          #{index + 1}
        </div>
      )}
      
      {/* Weekend Badge */}
      {isWeekend && (
        <div className="absolute top-4 right-4 bg-gradient-to-r from-blue-500 to-cyan-500 text-white text-xs font-bold px-2 py-1 rounded-full flex items-center gap-1">
          <Sun size={12} />
          Weekend
        </div>
      )}

      {/* Header */}
      <div className="mb-4">
        <div className="flex items-center text-sm text-gray-500 mb-2">
          <Calendar size={16} className="mr-2 text-indigo-500" />
          <span className="font-medium">{dayOfWeek}</span>
        </div>
        <h3 className="text-3xl font-black text-gray-900 mb-3 tracking-tight">{date}</h3>
        <ConfidenceBadge confidence={confidence} />
      </div>

      {/* Details */}
      <div className="space-y-3">
        <div className="flex items-center p-3 bg-indigo-50 rounded-xl transition-colors group-hover:bg-indigo-100">
          <Clock size={20} className="text-indigo-600 mr-3 flex-shrink-0" />
          <div>
            <span className="text-xs font-semibold text-indigo-600 uppercase tracking-wide">Maintenance Window</span>
            <p className="text-sm font-bold text-indigo-900">{window}</p>
          </div>
        </div>
        
        <div className="flex items-center p-3 bg-gray-50 rounded-xl transition-colors group-hover:bg-gray-100">
          <TrendingDown size={20} className="text-gray-600 mr-3 flex-shrink-0" />
          <div>
            <span className="text-xs font-semibold text-gray-600 uppercase tracking-wide">Expected Impact</span>
            <p className="text-sm font-bold text-gray-900">{impact}</p>
          </div>
        </div>
      </div>

      {/* Hover Effect Overlay */}
      <div className={`absolute inset-0 bg-gradient-to-br from-indigo-500/5 to-purple-500/5 rounded-2xl transition-opacity duration-300 pointer-events-none ${isHovered ? 'opacity-100' : 'opacity-0'}`} />
    </div>
  );
};

// --- Stats Card Component ---
const StatsCard = ({ icon: Icon, label, value, color, trend }) => {
  return (
    <div className="bg-white p-6 rounded-xl shadow-md border border-gray-100 hover:shadow-lg transition-all">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600 mb-1">{label}</p>
          <p className="text-3xl font-bold text-gray-900">{value}</p>
          {trend && <p className="text-xs text-green-600 font-semibold mt-1">↗ {trend}</p>}
        </div>
        <div className={`p-4 rounded-full ${color}`}>
          <Icon size={24} className="text-white" />
        </div>
      </div>
    </div>
  );
};

// --- Filter Component ---
const FilterPanel = ({ onFilterChange, activeFilter }) => {
  const filters = [
    { id: 'all', label: 'All Windows', icon: BarChart3 },
    { id: 'high', label: 'High Confidence', icon: CheckCircle },
    { id: 'weekend', label: 'Weekends Only', icon: Sun },
    { id: 'weekday', label: 'Weekdays Only', icon: Calendar }
  ];

  return (
    <div className="flex flex-wrap gap-2 mb-6">
      {filters.map(filter => (
        <button
          key={filter.id}
          onClick={() => onFilterChange(filter.id)}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-sm transition-all ${
            activeFilter === filter.id
              ? 'bg-indigo-600 text-white shadow-md'
              : 'bg-white text-gray-700 border border-gray-200 hover:border-indigo-300 hover:bg-indigo-50'
          }`}
        >
          <filter.icon size={16} />
          {filter.label}
        </button>
      ))}
    </div>
  );
};

// --- Enhanced Chatbot ---
const Chatbot = ({ predictions, apiBaseUrl }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [messages, setMessages] = useState([
    {
      type: 'bot',
      text: '👋 Hi! I\'m your Patch Window AI Assistant. I can help you understand the predictions and answer any questions. Try asking:\n\n• "What are the statistics?"\n• "What\'s the best window?"\n• "Show me weekend windows"',
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (isOpen && !isMinimized) {
      inputRef.current?.focus();
    }
  }, [isOpen, isMinimized]);

  const sendMessage = async () => {
    if (!inputValue.trim()) return;

    if (!predictions || predictions.length === 0) {
      setMessages(prev => [
        ...prev,
        {
          type: 'user',
          text: inputValue,
          timestamp: new Date()
        },
        {
          type: 'bot',
          text: '⚠️ No predictions available yet. Please generate predictions first by clicking "Generate Predictions".',
          timestamp: new Date()
        }
      ]);
      setInputValue('');
      return;
    }

    // Add user message
    setMessages(prev => [...prev, {
      type: 'user',
      text: inputValue,
      timestamp: new Date()
    }]);

    const userQuestion = inputValue;
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch(`${apiBaseUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: userQuestion,
          predictions: predictions
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();

      // Add bot response
      setMessages(prev => [
        ...prev,
        {
          type: 'bot',
          text: data.answer || data.error || 'No response from server',
          timestamp: new Date()
        }
      ]);
    } catch (error) {
      console.error('Chat Error:', error);
      setMessages(prev => [
        ...prev,
        {
          type: 'bot',
          text: '❌ Unable to connect to chatbot service. Please ensure the backend server is running at ' + apiBaseUrl,
          timestamp: new Date()
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <>
      {/* Floating Chat Button */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className="fixed bottom-6 right-6 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white px-4 py-3 rounded-full shadow-lg transition-all hover:scale-110 z-50 flex items-center gap-2"
          aria-label="Open chatbot"
        >
          <MessageCircle size={24} />
          <span className="text-sm font-medium hidden sm:inline">Ask AI</span>
        </button>
      )}

      {/* Chat Window */}
      {isOpen && (
        <div className={`fixed bottom-4 right-4 bg-white rounded-2xl shadow-2xl z-50 flex flex-col transition-all ${
          isMinimized ? 'w-80 h-16' : 'w-[95vw] sm:w-96 h-[80vh] sm:h-[600px]'
        } max-h-[calc(100vh-2rem)]`}>
          {/* Header */}
          <div className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white p-4 rounded-t-2xl flex items-center justify-between flex-shrink-0">
            <div className="flex items-center gap-2 min-w-0">
              <MessageCircle size={20} className="flex-shrink-0" />
              <div className="min-w-0">
                <h3 className="font-bold text-sm truncate">AI Assistant</h3>
                <p className="text-xs text-indigo-100 truncate">Powered by EPE Predictor</p>
              </div>
            </div>
            <div className="flex items-center gap-2 flex-shrink-0">
              <button
                onClick={() => setIsMinimized(!isMinimized)}
                className="hover:bg-indigo-700 p-1 rounded transition-colors"
                aria-label={isMinimized ? "Maximize" : "Minimize"}
              >
                {isMinimized ? <Maximize2 size={18} /> : <Minimize2 size={18} />}
              </button>
              <button
                onClick={() => setIsOpen(false)}
                className="hover:bg-indigo-700 p-1 rounded transition-colors"
                aria-label="Close chatbot"
              >
                <X size={18} />
              </button>
            </div>
          </div>

          {!isMinimized && (
            <>
              {/* Messages Area */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gradient-to-b from-indigo-50 to-white min-h-0">
                {messages.map((message, index) => (
                  <div
                    key={index}
                    className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[85%] p-3 rounded-lg ${
                        message.type === 'user'
                          ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-br-none'
                          : 'bg-white text-gray-800 shadow-md rounded-bl-none border-2 border-indigo-100'
                      }`}
                    >
                      <p className="text-sm whitespace-pre-line break-words">{message.text}</p>
                      <p className={`text-xs mt-1 ${
                        message.type === 'user' ? 'text-indigo-200' : 'text-gray-400'
                      }`}>
                        {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </p>
                    </div>
                  </div>
                ))}
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="bg-white p-3 rounded-lg shadow-md border-2 border-indigo-100">
                      <div className="flex items-center gap-2">
                        <RefreshCcw size={16} className="text-indigo-600 animate-spin" />
                        <span className="text-sm text-gray-600">Thinking...</span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Input Area */}
              <div className="p-4 bg-white border-t border-indigo-200 rounded-b-2xl flex-shrink-0">
                <div className="flex gap-2">
                  <input
                    ref={inputRef}
                    type="text"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Ask anything about predictions..."
                    className="flex-1 border-2 border-indigo-200 bg-white rounded-lg px-3 py-2 text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent min-w-0"
                    disabled={isLoading}
                  />
                  <button
                    onClick={sendMessage}
                    disabled={isLoading || !inputValue.trim()}
                    className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white p-2 rounded-lg hover:from-indigo-700 hover:to-purple-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex-shrink-0"
                    aria-label="Send message"
                  >
                    <Send size={18} />
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </>
  );
};

// --- Main App Component ---
const App = () => {
  const [predictions, setPredictions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [startDate, setStartDate] = useState(new Date().toISOString().split('T')[0]);
  const [numDays, setNumDays] = useState(30);
  const [activeFilter, setActiveFilter] = useState('all');

  const fetchPredictions = async () => {
    setIsLoading(true);
    setError('');

    try {
      const response = await fetch(`${API_BASE_URL}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          start_date: startDate,
          num_days: numDays
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.success) {
        setPredictions(data.predictions || []);
      } else {
        setError(data.error || 'Failed to fetch predictions');
      }
    } catch (err) {
      console.error('Fetch error:', err);
      setError(`Failed to connect to backend: ${err.message}. Make sure the Flask server is running at ${API_BASE_URL}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Filter predictions based on active filter
  const filteredPredictions = predictions.filter(prediction => {
    if (activeFilter === 'all') return true;
    if (activeFilter === 'high') return prediction.confidence >= 85;
    if (activeFilter === 'weekend') return ['Saturday', 'Sunday'].includes(prediction.dayOfWeek);
    if (activeFilter === 'weekday') return !['Saturday', 'Sunday'].includes(prediction.dayOfWeek);
    return true;
  });

  // Calculate statistics
  const stats = {
    total: predictions.length,
    highConfidence: predictions.filter(p => p.confidence >= 85).length,
    avgConfidence: predictions.length > 0 
      ? Math.round(predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length)
      : 0,
    weekendWindows: predictions.filter(p => ['Saturday', 'Sunday'].includes(p.dayOfWeek)).length
  };

  return (
    <div className="min-h-screen w-full font-sans overflow-x-hidden">
      {/* Background Pattern */}
      <div className="absolute inset-0 opacity-10 pointer-events-none">
        <div className="absolute inset-0" style={{
          backgroundImage: 'radial-gradient(circle at 2px 2px, white 1px, transparent 0)',
          backgroundSize: '40px 40px'
        }}></div>
      </div>

      <div className="relative max-w-7xl mx-auto w-full px-4 sm:px-8 py-8 pb-32">
        {/* Header with Gradient */}
        <header className="text-center mb-12 pt-8 animate-fadeIn">
          <div className="inline-block mb-4">
            <div className="bg-white/20 backdrop-blur-lg rounded-2xl px-4 py-2 inline-flex items-center gap-2">
              <Zap className="text-yellow-300" size={20} />
              <span className="text-white font-semibold text-sm">AI-Powered Predictions</span>
            </div>
          </div>
          <h1 className="text-5xl md:text-6xl font-black text-white mb-4 tracking-tight drop-shadow-lg">
            EPE Patch Window Predictor
          </h1>
          <p className="text-xl md:text-2xl text-white/90 font-medium max-w-2xl mx-auto">
            Identify high-confidence windows for system maintenance with AI-powered insights
          </p>
        </header>

        {/* Stats Cards */}
        {predictions.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8 animate-fadeIn">
            <StatsCard 
              icon={BarChart3}
              label="Total Windows"
              value={stats.total}
              color="bg-gradient-to-br from-blue-500 to-blue-600"
            />
            <StatsCard 
              icon={CheckCircle}
              label="High Confidence"
              value={stats.highConfidence}
              color="bg-gradient-to-br from-green-500 to-emerald-600"
            />
            <StatsCard 
              icon={TrendingDown}
              label="Avg. Confidence"
              value={`${stats.avgConfidence}%`}
              color="bg-gradient-to-br from-purple-500 to-purple-600"
            />
            <StatsCard 
              icon={Sun}
              label="Weekend Slots"
              value={stats.weekendWindows}
              color="bg-gradient-to-br from-orange-500 to-orange-600"
              trend="+12% optimal"
            />
          </div>
        )}

        {/* Input Controls Card */}
        <div className="bg-white/95 backdrop-blur-lg p-6 md:p-8 rounded-3xl shadow-2xl border border-white/20 mb-8 sticky top-4 z-10 animate-fadeIn">
          <div className="flex flex-col md:flex-row items-end space-y-4 md:space-y-0 md:space-x-4">
            
            <div className="flex-1 w-full">
              <label htmlFor="startDate" className="block text-sm font-bold text-gray-700 mb-2">Start Date</label>
              <input
                id="startDate"
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-full rounded-xl border-2 border-gray-300 bg-white shadow-sm focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500 p-3 transition-all text-gray-900 font-medium"
                style={{ colorScheme: 'light' }}
              />
            </div>

            <div className="flex-1 w-full">
              <label htmlFor="numDays" className="block text-sm font-bold text-gray-700 mb-2">Prediction Horizon (Days)</label>
              <input
                id="numDays"
                type="number"
                min="1"
                max="365"
                value={numDays}
                onChange={(e) => setNumDays(parseInt(e.target.value) || 1)}
                className="w-full rounded-xl border-2 border-gray-300 bg-white shadow-sm focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500 p-3 transition-all text-gray-900 font-medium"
                style={{ colorScheme: 'light' }}
              />
            </div>
            
            <button
              onClick={fetchPredictions}
              disabled={isLoading}
              className={`w-full md:w-auto flex items-center justify-center gap-3 px-8 py-3 border-0 text-base font-bold rounded-xl shadow-lg text-white transition-all transform active:scale-95 ${
                isLoading 
                  ? 'bg-gray-400 cursor-not-allowed' 
                  : 'bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 hover:shadow-xl'
              }`}
            >
              {isLoading ? (
                <>
                  <RefreshCcw size={20} className="animate-spin" />
                  <span>Generating...</span>
                </>
              ) : (
                <>
                  <Zap size={20} />
                  <span>Generate Predictions</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border-l-4 border-red-500 text-red-700 p-6 rounded-xl mb-8 shadow-lg animate-fadeIn" role="alert">
            <div className="flex items-start">
              <AlertTriangle className="mr-3 flex-shrink-0 mt-1" size={24} />
              <div>
                <p className="font-bold text-lg">Prediction Error</p>
                <p className="mt-1">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Loading State */}
        {isLoading && (
          <div className="text-center py-20 animate-fadeIn">
            <div className="inline-block p-8 bg-white/90 backdrop-blur-lg rounded-3xl shadow-2xl">
              <RefreshCcw size={48} className="text-indigo-600 animate-spin mx-auto mb-4" />
              <p className="text-gray-900 text-xl font-bold">Fetching optimal windows from AI...</p>
              <p className="text-gray-600 text-sm mt-2">Analyzing patterns and confidence levels</p>
            </div>
          </div>
        )}

        {/* Results Display */}
        {!isLoading && !error && predictions.length > 0 && (
          <div className="animate-fadeIn">
            {/* Filter Panel */}
            <FilterPanel onFilterChange={setActiveFilter} activeFilter={activeFilter} />

            <h2 className="text-3xl font-black text-white mb-6 flex items-center gap-3 drop-shadow-lg">
              <CheckCircle size={32} />
              Optimal Patch Windows ({filteredPredictions.length} Found)
            </h2>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 mb-24">
              {filteredPredictions.map((prediction, index) => (
                <PredictionCard key={index} prediction={prediction} index={index} />
              ))}
            </div>

            {filteredPredictions.length === 0 && (
              <div className="text-center py-12 bg-white/90 backdrop-blur-lg border-2 border-yellow-200 rounded-3xl">
                <AlertTriangle size={48} className="text-yellow-600 mx-auto mb-4" />
                <h2 className="text-2xl font-bold text-yellow-800 mb-2">No Windows Match Filter</h2>
                <p className="text-yellow-600">
                  Try selecting a different filter or adjusting your date range.
                </p>
              </div>
            )}
          </div>
        )}

        {!isLoading && !error && predictions.length === 0 && (
          <div className="text-center py-20 bg-white/90 backdrop-blur-lg border-2 border-gray-200 rounded-3xl animate-fadeIn">
            <Calendar size={64} className="text-gray-400 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-gray-800 mb-2">No Predictions Generated Yet</h2>
            <p className="text-gray-600 mb-6">
              Click "Generate Predictions" to find optimal maintenance windows
            </p>
          </div>
        )}

      </div>

      {/* Chatbot Component */}
      <Chatbot predictions={predictions} apiBaseUrl={API_BASE_URL} />
    </div>
  );
};

export default App;