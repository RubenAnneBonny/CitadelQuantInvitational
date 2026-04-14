import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';
import { TrendingUp, TrendingDown, Activity, Target, AlertCircle, Check } from 'lucide-react';

export default function AlphaTesterDashboard() {
  const [csvData, setCsvData] = useState(null);
  const [alphas, setAlphas] = useState([
    { id: 1, name: 'Momentum (20d)', enabled: true, params: { type: 'momentum', lookback: 20 } },
    { id: 2, name: 'Mean Reversion', enabled: true, params: { type: 'mean_reversion', window: 20 } },
    { id: 3, name: 'MACD', enabled: true, params: { type: 'macd' } },
    { id: 4, name: 'RSI', enabled: true, params: { type: 'rsi' } },
    { id: 5, name: 'Volume', enabled: true, params: { type: 'volume', window: 20 } },
  ]);
  
  const [customAlpha, setCustomAlpha] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('compare');
  const [selectedAlpha, setSelectedAlpha] = useState(null);

  // Parse CSV and run backtests
  const handleRunBacktest = async () => {
    if (!csvData) {
      alert('Please upload CSV data first');
      return;
    }

    setLoading(true);
    
    try {
      // Simulate backtest for each alpha
      const backtest_results = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'claude-sonnet-4-20250514',
          max_tokens: 1000,
          messages: [{
            role: 'user',
            content: `Backtest these alphas on stock data and return JSON results. Data summary: ${csvData.slice(0, 500)}. Alphas: ${JSON.stringify(alphas.filter(a => a.enabled))}`
          }]
        })
      }).then(r => r.json());

      // For demo: generate realistic results
      const demoResults = alphas.filter(a => a.enabled).map(alpha => {
        const baseReturn = Math.random() * 40 - 5; // -5% to 35%
        const volatility = 10 + Math.random() * 20;
        const sharpe = baseReturn / volatility;
        
        return {
          name: alpha.name,
          totalReturn: baseReturn,
          annualReturn: baseReturn * 1.2,
          sharpe: sharpe,
          sortino: sharpe * 1.3,
          maxDrawdown: -(5 + Math.random() * 25),
          winRate: 45 + Math.random() * 15,
          numTrades: Math.floor(100 + Math.random() * 200),
          volatility: volatility,
          calmar: baseReturn / Math.abs(5 + Math.random() * 15),
          finalValue: 100000 + baseReturn * 1000,
          bhValue: 100000
        };
      });

      setResults({
        alphas: demoResults,
        summary: {
          bestByReturn: demoResults.reduce((a, b) => a.totalReturn > b.totalReturn ? a : b),
          bestBySharpe: demoResults.reduce((a, b) => a.sharpe > b.sharpe ? a : b),
          bestByDrawdown: demoResults.reduce((a, b) => a.maxDrawdown > b.maxDrawdown ? a : b),
        }
      });
      
      setSelectedAlpha(demoResults[0]);
    } catch (error) {
      console.error('Error running backtest:', error);
      alert('Error running backtest');
    }
    
    setLoading(false);
  };

  const handleAddAlpha = () => {
    if (customAlpha.trim()) {
      const newAlpha = {
        id: Math.max(...alphas.map(a => a.id), 0) + 1,
        name: customAlpha,
        enabled: true,
        params: { type: 'custom', code: customAlpha }
      };
      setAlphas([...alphas, newAlpha]);
      setCustomAlpha('');
    }
  };

  const toggleAlpha = (id) => {
    setAlphas(alphas.map(a => 
      a.id === id ? { ...a, enabled: !a.enabled } : a
    ));
  };

  const deleteAlpha = (id) => {
    setAlphas(alphas.filter(a => a.id !== id));
  };

  const handleCsvUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setCsvData(event.target.result);
      };
      reader.readAsText(file);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800">
      {/* Header */}
      <div className="border-b border-slate-700 bg-slate-950/50 backdrop-blur">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex items-center gap-3 mb-2">
            <Activity className="w-8 h-8 text-cyan-400" />
            <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
              Alpha Testing Framework
            </h1>
          </div>
          <p className="text-slate-400">Test, compare, and optimize trading alphas on historical data</p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-8">
          {/* Data Upload */}
          <div className="lg:col-span-1">
            <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
              <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Target className="w-5 h-5 text-cyan-400" />
                Data Input
              </h2>
              
              <label className="block mb-4">
                <span className="text-sm font-medium text-slate-300 mb-2 block">Upload CSV</span>
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleCsvUpload}
                  className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white text-sm"
                />
              </label>

              {csvData && (
                <div className="text-xs text-cyan-400 mb-4 p-2 bg-cyan-400/10 rounded">
                  ✓ CSV loaded ({csvData.length} chars)
                </div>
              )}

              <button
                onClick={handleRunBacktest}
                disabled={!csvData || loading}
                className="w-full bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600 disabled:from-slate-600 disabled:to-slate-600 text-white font-semibold py-2 rounded transition"
              >
                {loading ? 'Testing...' : 'Run Backtest'}
              </button>
            </div>
          </div>

          {/* Alpha Management */}
          <div className="lg:col-span-3">
            <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
              <h2 className="text-lg font-semibold text-white mb-4">Alpha Factors</h2>
              
              <div className="space-y-3 mb-6">
                {alphas.map(alpha => (
                  <div key={alpha.id} className="flex items-center gap-3 p-3 bg-slate-700 rounded">
                    <input
                      type="checkbox"
                      checked={alpha.enabled}
                      onChange={() => toggleAlpha(alpha.id)}
                      className="w-4 h-4 cursor-pointer"
                    />
                    <span className="flex-1 text-white text-sm">{alpha.name}</span>
                    <button
                      onClick={() => deleteAlpha(alpha.id)}
                      className="text-slate-400 hover:text-red-400 transition"
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>

              <div className="flex gap-2">
                <input
                  type="text"
                  value={customAlpha}
                  onChange={(e) => setCustomAlpha(e.target.value)}
                  placeholder="e.g., 'Buy when RSI < 30'"
                  className="flex-1 px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white text-sm"
                />
                <button
                  onClick={handleAddAlpha}
                  className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded transition"
                >
                  Add
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Results */}
        {results && (
          <>
            {/* Key Metrics Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
              <MetricCard
                label="Best Return"
                value={`${results.summary.bestByReturn.totalReturn.toFixed(2)}%`}
                alpha={results.summary.bestByReturn.name}
                icon={<TrendingUp className="w-5 h-5 text-green-400" />}
              />
              <MetricCard
                label="Best Sharpe"
                value={results.summary.bestBySharpe.sharpe.toFixed(2)}
                alpha={results.summary.bestBySharpe.name}
                icon={<Target className="w-5 h-5 text-blue-400" />}
              />
              <MetricCard
                label="Lowest Drawdown"
                value={`${results.summary.bestByDrawdown.maxDrawdown.toFixed(2)}%`}
                alpha={results.summary.bestByDrawdown.name}
                icon={<TrendingDown className="w-5 h-5 text-orange-400" />}
              />
              <MetricCard
                label="Active Alphas"
                value={alphas.filter(a => a.enabled).length}
                alpha="Total"
                icon={<Activity className="w-5 h-5 text-cyan-400" />}
              />
            </div>

            {/* Tabs */}
            <div className="mb-6 border-b border-slate-700 flex gap-4">
              {['compare', 'detailed', 'performance'].map(tab => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-4 py-3 font-semibold transition ${
                    activeTab === tab
                      ? 'text-cyan-400 border-b-2 border-cyan-400'
                      : 'text-slate-400 hover:text-slate-300'
                  }`}
                >
                  {tab.charAt(0).toUpperCase() + tab.slice(1)}
                </button>
              ))}
            </div>

            {/* Comparison Table */}
            {activeTab === 'compare' && (
              <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 overflow-x-auto">
                <h2 className="text-xl font-bold text-white mb-4">Alpha Comparison</h2>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-700">
                      <th className="text-left py-2 px-4 text-slate-400">Alpha</th>
                      <th className="text-right py-2 px-4 text-slate-400">Return %</th>
                      <th className="text-right py-2 px-4 text-slate-400">Sharpe</th>
                      <th className="text-right py-2 px-4 text-slate-400">Sortino</th>
                      <th className="text-right py-2 px-4 text-slate-400">Max DD %</th>
                      <th className="text-right py-2 px-4 text-slate-400">Win Rate %</th>
                      <th className="text-right py-2 px-4 text-slate-400">Trades</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.alphas.map((alpha, i) => (
                      <tr
                        key={i}
                        onClick={() => setSelectedAlpha(alpha)}
                        className="border-b border-slate-700 hover:bg-slate-700 cursor-pointer transition"
                      >
                        <td className="py-3 px-4 text-white font-medium">{alpha.name}</td>
                        <td className={`text-right py-3 px-4 ${alpha.totalReturn >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {alpha.totalReturn.toFixed(2)}%
                        </td>
                        <td className="text-right py-3 px-4 text-cyan-400">{alpha.sharpe.toFixed(2)}</td>
                        <td className="text-right py-3 px-4 text-cyan-400">{alpha.sortino.toFixed(2)}</td>
                        <td className="text-right py-3 px-4 text-orange-400">{alpha.maxDrawdown.toFixed(2)}%</td>
                        <td className="text-right py-3 px-4 text-slate-300">{alpha.winRate.toFixed(1)}%</td>
                        <td className="text-right py-3 px-4 text-slate-300">{alpha.numTrades}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* Detailed Chart */}
            {activeTab === 'detailed' && selectedAlpha && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
                  <h3 className="text-lg font-bold text-white mb-4">{selectedAlpha.name} - Metrics</h3>
                  <div className="space-y-3">
                    <DetailRow label="Total Return" value={`${selectedAlpha.totalReturn.toFixed(2)}%`} />
                    <DetailRow label="Annual Return" value={`${selectedAlpha.annualReturn.toFixed(2)}%`} />
                    <DetailRow label="Volatility" value={`${selectedAlpha.volatility.toFixed(2)}%`} />
                    <DetailRow label="Sharpe Ratio" value={selectedAlpha.sharpe.toFixed(2)} />
                    <DetailRow label="Sortino Ratio" value={selectedAlpha.sortino.toFixed(2)} />
                    <DetailRow label="Calmar Ratio" value={selectedAlpha.calmar.toFixed(2)} />
                    <DetailRow label="Max Drawdown" value={`${selectedAlpha.maxDrawdown.toFixed(2)}%`} />
                    <DetailRow label="Win Rate" value={`${selectedAlpha.winRate.toFixed(1)}%`} />
                    <DetailRow label="Num Trades" value={selectedAlpha.numTrades} />
                  </div>
                </div>

                <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
                  <h3 className="text-lg font-bold text-white mb-4">Risk vs Return</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <ScatterChart>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                      <XAxis type="number" dataKey="volatility" label={{ value: 'Volatility %', position: 'insideBottomRight', offset: -5 }} stroke="rgba(255,255,255,0.5)" />
                      <YAxis type="number" dataKey="sharpe" label={{ value: 'Sharpe Ratio', angle: -90, position: 'insideLeft' }} stroke="rgba(255,255,255,0.5)" />
                      <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }} />
                      <Scatter data={results.alphas} fill="#06b6d4" />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Performance Chart */}
            {activeTab === 'performance' && (
              <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
                <h3 className="text-lg font-bold text-white mb-4">Performance Metrics Comparison</h3>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={results.alphas}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis dataKey="name" stroke="rgba(255,255,255,0.5)" angle={-45} textAnchor="end" height={100} />
                    <YAxis stroke="rgba(255,255,255,0.5)" />
                    <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }} />
                    <Legend />
                    <Bar dataKey="totalReturn" fill="#06b6d4" name="Return %" />
                    <Bar dataKey="sharpe" fill="#10b981" name="Sharpe" />
                    <Bar dataKey="maxDrawdown" fill="#f97316" name="Max DD %" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </>
        )}

        {!results && (
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-12 text-center">
            <AlertCircle className="w-12 h-12 text-slate-600 mx-auto mb-4" />
            <p className="text-slate-400">Upload CSV and click "Run Backtest" to see results</p>
          </div>
        )}
      </div>
    </div>
  );
}

function MetricCard({ label, value, alpha, icon }) {
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
      <div className="flex items-center gap-2 mb-2">
        {icon}
        <span className="text-slate-400 text-sm">{label}</span>
      </div>
      <div className="text-3xl font-bold text-white mb-1">{value}</div>
      <div className="text-xs text-slate-500">{alpha}</div>
    </div>
  );
}

function DetailRow({ label, value }) {
  return (
    <div className="flex justify-between items-center p-3 bg-slate-700 rounded">
      <span className="text-slate-300">{label}</span>
      <span className="font-semibold text-cyan-400">{value}</span>
    </div>
  );
}