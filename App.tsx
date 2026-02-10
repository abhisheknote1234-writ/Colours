
import React, { useState, useEffect, useCallback } from 'react';
import Layout from './components/Layout';
import StateCard from './components/StateCard';
import LiveMetrics from './components/LiveMetrics';
import ChartsSection from './components/ChartsSection';
import SupportSection from './components/SupportSection';
import AlertOverlay from './components/AlertOverlay';
import { StudyData, HistoryPoint } from './types';
import { fetchRealtimeData } from './services/mockDataService';

const App: React.FC = () => {
  const [currentData, setCurrentData] = useState<StudyData | null>(null);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [showAlert, setShowAlert] = useState(false);
  const [view, setView] = useState<'home' | 'details'>('home');

  const updateData = useCallback(async () => {
    try {
      const newData = await fetchRealtimeData();
      setCurrentData(newData);
      setIsConnected(true);
      
      setHistory(prev => {
        const newPoint = {
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
          heart_rate: newData.heart_rate,
          hrv: newData.hrv,
          learning_capacity: newData.learning_capacity
        };
        const updated = [...prev, newPoint];
        return updated.slice(-30); // Keep last 30 data points
      });

      // Logic to trigger alert
      if (['stress', 'fatigue', 'low_learning'].includes(newData.state) && newData.learning_capacity < 50) {
        setShowAlert(true);
      } else {
        setShowAlert(false);
      }
    } catch (err) {
      setIsConnected(false);
    }
  }, []);

  useEffect(() => {
    const timer = setInterval(updateData, 2000);
    return () => clearInterval(timer);
  }, [updateData]);

  if (!currentData) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50">
        <div className="text-center animate-pulse">
          <div className="w-16 h-16 border-4 border-blue-200 border-t-blue-600 rounded-full mx-auto mb-4 animate-spin" />
          <p className="text-slate-500 font-medium">Connecting to Sensor...</p>
        </div>
      </div>
    );
  }

  return (
    <Layout connected={isConnected}>
      {/* View Switcher */}
      <div className="flex bg-white p-1 rounded-xl shadow-sm border border-slate-100 w-fit self-center sm:self-end">
        <button 
          onClick={() => setView('home')}
          className={`px-6 py-2 rounded-lg text-sm font-bold transition-all ${view === 'home' ? 'bg-slate-900 text-white shadow-md' : 'text-slate-500 hover:bg-slate-50'}`}
        >
          Dashboard
        </button>
        <button 
          onClick={() => setView('details')}
          className={`px-6 py-2 rounded-lg text-sm font-bold transition-all ${view === 'details' ? 'bg-slate-900 text-white shadow-md' : 'text-slate-500 hover:bg-slate-50'}`}
        >
          Detailed Trends
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 flex flex-col gap-6">
          {view === 'home' ? (
            <>
              <StateCard data={currentData} />
              <LiveMetrics data={currentData} />
              <div className="bg-blue-600 text-white p-6 rounded-2xl shadow-lg relative overflow-hidden group">
                <div className="relative z-10">
                  <h3 className="text-sm font-bold uppercase tracking-widest opacity-80 mb-2">Guidance Focus</h3>
                  <p className="text-xl font-medium leading-relaxed">
                    “{currentData.message}”
                  </p>
                </div>
                {/* Decorative background circle */}
                <div className="absolute -right-12 -bottom-12 w-48 h-48 bg-white opacity-5 rounded-full" />
              </div>
            </>
          ) : (
            <ChartsSection history={history} />
          )}
        </div>

        <div className="flex flex-col gap-6">
          <SupportSection data={currentData} />
          
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
             <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4">Study Session Stats</h4>
             <div className="space-y-4">
               <div className="flex justify-between items-center">
                 <span className="text-sm text-slate-600">Session Time</span>
                 <span className="text-sm font-bold text-slate-800">1h 12m</span>
               </div>
               <div className="flex justify-between items-center">
                 <span className="text-sm text-slate-600">Peak Flow Reach</span>
                 <span className="text-sm font-bold text-emerald-600">22 min</span>
               </div>
               <div className="flex justify-between items-center">
                 <span className="text-sm text-slate-600">Regulation Score</span>
                 <span className="text-sm font-bold text-slate-800">84/100</span>
               </div>
             </div>
          </div>
        </div>
      </div>

      <AlertOverlay 
        data={currentData} 
        visible={showAlert} 
        onClose={() => setShowAlert(false)} 
      />
    </Layout>
  );
};

export default App;
