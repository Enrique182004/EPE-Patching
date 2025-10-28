import React, { useState, useEffect, useRef } from 'react';
import { MessageCircle, Send, X, Minimize2, Maximize2, Sparkles, Zap } from 'lucide-react';

/**
 * FREE AI-Powered Chatbot for Google Colab
 * Uses Google Gemini API (completely FREE)
 * 
 * Features:
 * - Explains why certain windows are better
 * - Compares windows intelligently  
 * - Provides best practices
 * - Pattern analysis
 */

const Chatbot = ({ predictions, apiBaseUrl }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [messages, setMessages] = useState([
    {
      type: 'bot',
      text: '👋 Hi! I\'m your FREE AI Assistant powered by Google Gemini.\n\nI can help you:\n• Understand WHY certain windows are recommended\n• Explain the reasoning behind predictions\n• Compare different maintenance windows\n• Analyze patterns in your data\n• Provide best practices\n\nTry asking:\n• "Why is the first window better than the second?"\n• "Why are weekends better?"\n• "Explain why #1 was chosen"\n• "What makes overnight hours optimal?"',
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

  /**
   * Send question to FREE Gemini AI
   */
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
      // Call FREE Gemini AI endpoint
      const response = await fetch(`${apiBaseUrl}/api/ai-explain`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true'
        },
        body: JSON.stringify({
          question: userQuestion,
          predictions: predictions
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();

      // Add AI response
      setMessages(prev => [
        ...prev,
        {
          type: 'bot',
          text: data.answer,
          timestamp: new Date(),
          isAI: true,
          source: data.source
        }
      ]);
    } catch (error) {
      console.error('AI Error:', error);
      setMessages(prev => [
        ...prev,
        {
          type: 'bot',
          text: '❌ Unable to connect to AI service. Please ensure:\n\n1. Your Colab notebook is running\n2. You have a valid Gemini API key\n3. The ngrok URL is correct in App.jsx',
          timestamp: new Date()
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Compare two windows
   */
  const compareWindows = async (index1, index2) => {
    setIsLoading(true);
    
    try {
      const response = await fetch(`${apiBaseUrl}/api/ai-compare`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true'
        },
        body: JSON.stringify({
          index1: index1,
          index2: index2,
          predictions: predictions
        })
      });

      const data = await response.json();

      setMessages(prev => [
        ...prev,
        {
          type: 'bot',
          text: `🔍 **Window Comparison:**\n\n${data.comparison}`,
          timestamp: new Date(),
          isAI: true
        }
      ]);
    } catch (error) {
      console.error('Comparison Error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Get best practices
   */
  const getBestPractices = async () => {
    setIsLoading(true);
    
    setMessages(prev => [
      ...prev,
      {
        type: 'user',
        text: 'Show me best practices',
        timestamp: new Date()
      }
    ]);

    try {
      const response = await fetch(`${apiBaseUrl}/api/ai-best-practices`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true'
        },
        body: JSON.stringify({
          predictions: predictions
        })
      });

      const data = await response.json();

      setMessages(prev => [
        ...prev,
        {
          type: 'bot',
          text: `📋 **Best Practices Analysis:**\n\n${data.best_practices}`,
          timestamp: new Date(),
          isAI: true
        }
      ]);
    } catch (error) {
      console.error('Best Practices Error:', error);
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

  const quickActions = [
    { 
      label: "Why is #1 the best?", 
      action: () => {
        setInputValue("Why is the top-ranked window the best choice? Explain the reasoning.");
        setTimeout(() => sendMessage(), 100);
      }
    },
    { 
      label: "Compare #1 vs #2", 
      action: () => compareWindows(0, 1)
    },
    { 
      label: "Why weekends better?", 
      action: () => {
        setInputValue("Why are weekend windows better than weekdays?");
        setTimeout(() => sendMessage(), 100);
      }
    },
    { 
      label: "Overnight hours?", 
      action: () => {
        setInputValue("Why are overnight hours optimal for maintenance?");
        setTimeout(() => sendMessage(), 100);
      }
    }
  ];

  return (
    <>
      {/* Floating Chat Button */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className="fixed bottom-6 right-6 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white px-4 py-3 rounded-full shadow-lg transition-all hover:scale-110 z-50 flex items-center gap-2"
          aria-label="Open AI chatbot"
        >
          <Sparkles size={24} />
          <span className="text-sm font-medium hidden sm:inline">Free AI Help</span>
        </button>
      )}

      {/* Chat Window */}
      {isOpen && (
        <div className={`fixed bottom-4 right-4 bg-white rounded-2xl shadow-2xl z-50 flex flex-col transition-all ${
          isMinimized ? 'w-80 h-16' : 'w-[95vw] sm:w-96 h-[80vh] sm:h-[600px]'
        } max-h-[calc(100vh-2rem)]`}>
          {/* Header */}
          <div className="bg-gradient-to-r from-purple-600 to-pink-600 text-white p-4 rounded-t-2xl flex items-center justify-between flex-shrink-0">
            <div className="flex items-center gap-2 min-w-0">
              <Sparkles size={20} className="flex-shrink-0 animate-pulse" />
              <div className="min-w-0">
                <h3 className="font-bold text-sm truncate">AI Assistant</h3>
                <p className="text-xs text-purple-100 truncate">Powered by Gemini (FREE)</p>
              </div>
            </div>
            <div className="flex items-center gap-2 flex-shrink-0">
              <button
                onClick={() => setIsMinimized(!isMinimized)}
                className="hover:bg-purple-700 p-1 rounded transition-colors"
                aria-label={isMinimized ? "Maximize" : "Minimize"}
              >
                {isMinimized ? <Maximize2 size={18} /> : <Minimize2 size={18} />}
              </button>
              <button
                onClick={() => setIsOpen(false)}
                className="hover:bg-purple-700 p-1 rounded transition-colors"
                aria-label="Close chatbot"
              >
                <X size={18} />
              </button>
            </div>
          </div>

          {!isMinimized && (
            <>
              {/* Messages Area */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gradient-to-b from-purple-50 to-white min-h-0">
                {messages.map((message, index) => (
                  <div
                    key={index}
                    className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[85%] p-3 rounded-lg ${
                        message.type === 'user'
                          ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-br-none'
                          : 'bg-white text-gray-800 shadow-md rounded-bl-none border-2 border-purple-100'
                      }`}
                    >
                      {message.isAI && (
                        <div className="flex items-center gap-1 mb-2 text-purple-600">
                          <Sparkles size={14} />
                          <span className="text-xs font-semibold">AI Response</span>
                        </div>
                      )}
                      <p className="text-sm whitespace-pre-line break-words">{message.text}</p>
                      <p className={`text-xs mt-1 ${
                        message.type === 'user' ? 'text-purple-200' : 'text-gray-400'
                      }`}>
                        {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </p>
                    </div>
                  </div>
                ))}
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="bg-white p-3 rounded-lg shadow-md border-2 border-purple-100">
                      <div className="flex items-center gap-2">
                        <Sparkles size={16} className="text-purple-600 animate-pulse" />
                        <span className="text-sm text-gray-600">AI is thinking...</span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Quick Actions */}
              {messages.length === 1 && predictions.length > 0 && (
                <div className="p-3 bg-gradient-to-r from-purple-50 to-pink-50 border-t border-purple-200 flex-shrink-0">
                  <p className="text-xs text-purple-700 font-semibold mb-2">✨ Quick Actions:</p>
                  <div className="grid grid-cols-2 gap-2">
                    {quickActions.map((action, index) => (
                      <button
                        key={index}
                        onClick={action.action}
                        className="text-xs bg-white text-purple-700 px-3 py-2 rounded-lg hover:bg-purple-100 transition-colors border border-purple-200 font-medium"
                      >
                        {action.label}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Input Area */}
              <div className="p-4 bg-white border-t border-purple-200 rounded-b-2xl flex-shrink-0">
                <div className="flex gap-2">
                  <input
                    ref={inputRef}
                    type="text"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Ask AI anything about predictions..."
                    className="flex-1 border-2 border-purple-200 bg-white rounded-lg px-3 py-2 text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent min-w-0"
                    disabled={isLoading}
                  />
                  <button
                    onClick={sendMessage}
                    disabled={isLoading || !inputValue.trim()}
                    className="bg-gradient-to-r from-purple-600 to-pink-600 text-white p-2 rounded-lg hover:from-purple-700 hover:to-pink-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex-shrink-0"
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

export default Chatbot;