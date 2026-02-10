
import React from 'react';
import { StudyData } from '../types';

interface Props {
  data: StudyData;
  visible: boolean;
  onClose: () => void;
}

const AlertOverlay: React.FC<Props> = ({ data, visible, onClose }) => {
  if (!visible) return null;

  return (
    <div className="fixed bottom-6 left-1/2 -translate-x-1/2 w-full max-w-md px-4 z-50 animate-bounce-in">
      <div className="bg-rose-600 text-white p-5 rounded-2xl shadow-2xl flex flex-col gap-3">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <h4 className="font-bold">Learning Strain Detected</h4>
          </div>
          <button onClick={onClose} className="p-1 hover:bg-rose-500 rounded-lg transition-colors">
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
          </button>
        </div>
        
        <p className="text-sm opacity-90 leading-relaxed">
          It looks like learning is becoming harder right now. This is normal. Your regulation signals show a need for a shift in approach.
        </p>
        
        <div className="flex gap-2 mt-1">
          <button className="flex-1 bg-white text-rose-600 px-4 py-2 rounded-xl text-xs font-bold hover:bg-rose-50 transition-colors">
            Take 5m Break
          </button>
          <button className="flex-1 bg-rose-500 text-white border border-rose-400 px-4 py-2 rounded-xl text-xs font-bold hover:bg-rose-400 transition-colors">
            Change Topic
          </button>
        </div>
      </div>
    </div>
  );
};

export default AlertOverlay;
