
import React from 'react';
import { StudyData } from '../types';

interface Props {
  data: StudyData;
}

const LiveMetrics: React.FC<Props> = ({ data }) => {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
      <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-100 flex items-center justify-between">
        <div>
          <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Heart Rate</p>
          <div className="flex items-baseline gap-1">
            <span className="text-4xl font-bold text-slate-800 leading-none">{data.heart_rate}</span>
            <span className="text-slate-400 text-sm font-medium">BPM</span>
          </div>
        </div>
        <div className="w-12 h-12 flex items-center justify-center bg-rose-50 rounded-full">
           <svg className="w-6 h-6 text-rose-500 pulse" fill="currentColor" viewBox="0 0 24 24">
             <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>
           </svg>
        </div>
      </div>

      <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-100 flex items-center justify-between">
        <div>
          <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1">Heart Variability (HRV)</p>
          <div className="flex items-baseline gap-1">
            <span className="text-4xl font-bold text-slate-800 leading-none">{data.hrv}</span>
            <span className="text-slate-400 text-sm font-medium">ms</span>
          </div>
        </div>
        <div className="w-12 h-12 flex items-center justify-center bg-blue-50 rounded-full">
           <svg className="w-6 h-6 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="2.5">
             <path strokeLinecap="round" strokeLinejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
           </svg>
        </div>
      </div>
    </div>
  );
};

export default LiveMetrics;
