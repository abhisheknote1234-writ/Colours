
import React from 'react';
import { LearningState } from './types';

export const STATE_CONFIG: Record<LearningState, { label: string; color: string; bgColor: string; description: string }> = {
  flow: { 
    label: 'High Learning State (Flow)', 
    color: 'text-emerald-700', 
    bgColor: 'bg-emerald-50',
    description: 'You are deeply focused and absorbing information efficiently.'
  },
  neutral: { 
    label: 'Neutral State', 
    color: 'text-blue-700', 
    bgColor: 'bg-blue-50',
    description: 'Steady progress. You are maintaining a consistent learning pace.'
  },
  stress: { 
    label: 'Learning Stress', 
    color: 'text-rose-700', 
    bgColor: 'bg-rose-50',
    description: 'Your brain might be feeling overloaded. Consider a short pause.'
  },
  fatigue: { 
    label: 'Mental Fatigue', 
    color: 'text-amber-700', 
    bgColor: 'bg-amber-50',
    description: 'Focus is dipping. A 5-minute break could reset your energy.'
  },
  distraction: { 
    label: 'Distraction Detected', 
    color: 'text-indigo-700', 
    bgColor: 'bg-indigo-50',
    description: 'Attention seems to be drifting away from the core topic.'
  },
  loss_of_interest: { 
    label: 'Loss of Interest', 
    color: 'text-slate-700', 
    bgColor: 'bg-slate-50',
    description: 'The current material may not be engaging you effectively.'
  },
  low_learning: { 
    label: 'Low Learning State', 
    color: 'text-orange-700', 
    bgColor: 'bg-orange-50',
    description: 'Cognitive load is high, but retention might be lower right now.'
  },
};

export const MOCK_TOPICS = [
  "Laplace Transform",
  "Neural Networks",
  "Thermodynamics",
  "Linear Algebra",
  "Organic Chemistry"
];
