
import React from 'react';
import { StudyData } from '../types';
import { STATE_CONFIG } from '../constants';

interface Props {
  data: StudyData;
}

const StateCard: React.FC<Props> = ({ data }) => {
  const config = STATE_CONFIG[data.state];
  
  return (
    <div className={`p-6 rounded-3xl ${config.bgColor} border border-opacity-20 flex flex-col md:flex-row gap-6 items-center transition-all duration-500`}>
      <div className="flex-1 text-center md:text-left">
        <span className={`text-xs font-bold uppercase tracking-widest ${config.color} mb-1 block`}>Current State</span>
        <h2 className={`text-3xl font-bold ${config.color} mb-2`}>{config.label}</h2>
        <p className="text-slate-600 font-medium leading-relaxed">
          {config.description}
        </p>
      </div>
      
      <div className="flex flex-col items-center justify-center p-6 bg-white rounded-2xl shadow-sm min-w-[160px]">
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Capacity</span>
        <div className="relative w-24 h-24 flex items-center justify-center">
          <svg className="w-full h-full transform -rotate-90">
            <circle
              cx="48" cy="48" r="40"
              fill="transparent"
              stroke="#f1f5f9"
              strokeWidth="8"
            />
            <circle
              cx="48" cy="48" r="40"
              fill="transparent"
              stroke="currentColor"
              strokeWidth="8"
              strokeDasharray={251.2}
              strokeDashoffset={251.2 * (1 - data.learning_capacity / 100)}
              className={`${config.color} transition-all duration-700 ease-out`}
            />
          </svg>
          <span className={`absolute text-2xl font-bold ${config.color}`}>
            {data.learning_capacity}%
          </span>
        </div>
      </div>
    </div>
  );
};

export default StateCard;
