
import React from 'react';
import { StudyData, LearningSuggestion } from '../types';

interface Props {
  data: StudyData;
}

const SupportSection: React.FC<Props> = ({ data }) => {
  const suggestions: LearningSuggestion[] = [
    { 
      type: 'simpler', 
      title: 'Simple Analogy', 
      content: `Let's look at ${data.topic} as if it were a water pipe system. Imagine the flow is controlled by...` 
    },
    { 
      type: 'summary', 
      title: 'Quick Summary', 
      content: `In essence, ${data.topic} allows us to transform time-domain problems into simpler frequency-domain algebra.` 
    },
    { 
      type: 'key_concepts', 
      title: 'Key Concepts Only', 
      content: `Focus on: 1. Linearity 2. Time Shifting 3. Differentiation. Ignore the complex derivations for now.` 
    },
  ];

  return (
    <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-bold text-slate-800">Smart Support</h3>
          <p className="text-sm text-slate-500">Tailored to: <span className="text-blue-600 font-semibold">{data.topic}</span></p>
        </div>
        <div className="bg-blue-100 p-2 rounded-lg">
           <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
             <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
           </svg>
        </div>
      </div>

      <div className="space-y-4">
        {suggestions.map((item, idx) => (
          <button 
            key={idx}
            className="w-full text-left p-4 rounded-xl border border-slate-100 hover:border-blue-200 hover:bg-blue-50 transition-all group"
          >
            <h4 className="text-sm font-bold text-slate-700 group-hover:text-blue-700 mb-1">{item.title}</h4>
            <p className="text-xs text-slate-500 leading-relaxed line-clamp-2">{item.content}</p>
          </button>
        ))}
      </div>
      
      <div className="mt-6 p-4 bg-slate-50 rounded-xl border border-slate-200">
        <p className="text-sm text-slate-600 italic">
          "{data.message}"
        </p>
      </div>
    </div>
  );
};

export default SupportSection;
