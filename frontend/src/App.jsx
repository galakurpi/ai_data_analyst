import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';

function App() {
  const [messages, setMessages] = useState([
    { role: 'ai', content: 'Hello! I am your Data Analyst. Ask me about your sales data.' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const startNewChat = () => {
    setConversationId(null);
    setMessages([{ role: 'ai', content: 'Hello! I am your Data Analyst. Ask me about your sales data.' }]);
  };

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMsg = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const response = await axios.post('/api/chat', {
        query: input,
        conversation_id: conversationId
      });
      const data = response.data;

      if (data.conversation_id) {
        setConversationId(data.conversation_id);
      }

      const aiMsg = {
        role: 'ai',
        content: data.summary,
        sql: data.sql,
        data_structure: data.data_structure,
        plotly_json: data.plotly_json ? JSON.parse(data.plotly_json) : null
      };

      setMessages(prev => [...prev, aiMsg]);
    } catch (error) {
      console.error("Error:", error);
      const errorMsg = {
        role: 'ai',
        content: "Sorry, something went wrong. Please try again.",
        error: error.response?.data?.error || error.message
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') sendMessage();
  };

  return (
    <div className="flex flex-col h-screen bg-zinc-950 text-zinc-100 font-sans selection:bg-blue-500/30">
      {/* Header */}
      <header className="border-b border-zinc-800 bg-zinc-900/50 backdrop-blur-md p-4 sticky top-0 z-10">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
              <span className="text-white font-bold text-lg">AI</span>
            </div>
            <h1 className="text-lg font-semibold tracking-tight text-zinc-100">
              Data Analyst
            </h1>
          </div>
          <button
            onClick={startNewChat}
            className="text-sm bg-zinc-800 hover:bg-zinc-700 text-zinc-300 px-3 py-1.5 rounded-md transition-colors border border-zinc-700"
          >
            + New Chat
          </button>
        </div>
      </header>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6 scroll-smooth">
        <div className="max-w-5xl mx-auto space-y-6">
          {messages.map((msg, idx) => (
            <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-2 duration-300`}>
              <div className={`max-w-[90%] sm:max-w-[80%] rounded-2xl p-5 shadow-sm ${msg.role === 'user'
                ? 'bg-blue-600 text-white rounded-tr-sm shadow-blue-900/20'
                : 'bg-zinc-900 border border-zinc-800 text-zinc-300 rounded-tl-sm shadow-zinc-900/50'
                }`}>
                <p className="whitespace-pre-wrap leading-relaxed">{msg.content}</p>

                {/* Visualization */}
                {msg.plotly_json && (
                  <div className="mt-5 bg-zinc-950 p-1 rounded-xl border border-zinc-800 w-full overflow-hidden shadow-inner">
                    <Plot
                      data={msg.plotly_json.data}
                      layout={{
                        ...msg.plotly_json.layout,
                        autosize: true,
                        width: undefined,
                        height: undefined,
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        font: { color: '#e4e4e7' }, // zinc-200
                        margin: { l: 50, r: 50, t: 50, b: 50 },
                        xaxis: { ...msg.plotly_json.layout.xaxis, gridcolor: '#27272a' }, // zinc-800
                        yaxis: { ...msg.plotly_json.layout.yaxis, gridcolor: '#27272a' },
                      }}
                      useResizeHandler={true}
                      style={{ width: "100%", minHeight: "450px", minWidth: "600px" }}
                      className="w-full h-full"
                      config={{ responsive: true, displayModeBar: true }}
                    />
                  </div>
                )}

                {/* Debug/SQL Section */}
                {(msg.sql || msg.error) && (
                  <details className="mt-3 group">
                    <summary className={`text-xs font-medium cursor-pointer opacity-60 hover:opacity-100 transition-opacity flex items-center gap-1 ${msg.role === 'user' ? 'text-blue-200' : 'text-zinc-500'}`}>
                      <span className="group-open:rotate-90 transition-transform">▶</span> Debug Info
                    </summary>
                    <div className={`p-3 rounded-2xl text-sm shadow-sm ${msg.role === 'user'
                        ? 'bg-blue-600 text-white rounded-tr-none'
                        : 'bg-zinc-800 text-zinc-200 rounded-tl-none border border-zinc-700'
                      }`}>
                      {msg.content.replace(/\\\$/g, '$')}
                    </div>
                    <div className={`mt-2 p-3 rounded-lg text-xs font-mono overflow-x-auto border ${msg.role === 'user'
                      ? 'bg-blue-700/50 border-blue-600/50 text-blue-100'
                      : 'bg-zinc-950 border-zinc-800 text-zinc-400'}`}>
                      {msg.sql && (
                        <div className="mb-2">
                          <strong className="block mb-1 opacity-70">SQL Generated:</strong>
                          <code className="block whitespace-pre-wrap break-all">{msg.sql}</code>
                        </div>
                      )}
                      {msg.error && (
                        <div className="text-red-400">
                          <strong className="block mb-1">Error:</strong>
                          {msg.error}
                        </div>
                      )}
                      {msg.data_structure && (
                        <div className="mt-3">
                          <strong className="block mb-1 opacity-70">Response Structure:</strong>
                          <div className="text-xs text-zinc-500 mb-2">Total Rows: {msg.data_structure.rows}</div>
                          <div className="overflow-x-auto">
                            <table className="min-w-full text-xs text-left border-collapse">
                              <thead>
                                <tr className="border-b border-zinc-700">
                                  <th className="p-2 font-semibold text-zinc-300">Column Name</th>
                                  <th className="p-2 font-semibold text-zinc-300">Data Type</th>
                                </tr>
                              </thead>
                              <tbody>
                                {msg.data_structure.columns.map((col, i) => (
                                  <tr key={i} className="border-b border-zinc-800 last:border-0 hover:bg-zinc-900/50">
                                    <td className="p-2 text-zinc-400 font-mono">{col.name}</td>
                                    <td className="p-2 text-zinc-500 font-mono">{col.type}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}
                    </div>
                  </details>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex justify-start animate-pulse">
              <div className="bg-zinc-900 border border-zinc-800 rounded-2xl rounded-tl-sm p-5 shadow-sm">
                <div className="flex gap-2 items-center h-6">
                  <div className="w-2 h-2 bg-zinc-500 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-zinc-500 rounded-full animate-bounce delay-75"></div>
                  <div className="w-2 h-2 bg-zinc-500 rounded-full animate-bounce delay-150"></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="p-4 bg-zinc-900/80 backdrop-blur-xl border-t border-zinc-800">
        <div className="max-w-5xl mx-auto flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask a question about your data..."
            className="flex-1 p-4 bg-zinc-800/50 border border-zinc-700 text-zinc-100 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 placeholder-zinc-500 transition-all shadow-inner"
            disabled={loading}
          />
          <button
            onClick={sendMessage}
            disabled={loading || !input.trim()}
            className="bg-gradient-to-r from-blue-600 to-blue-500 text-white px-8 py-4 rounded-xl font-semibold hover:from-blue-500 hover:to-blue-400 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-blue-900/20 active:scale-95"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
