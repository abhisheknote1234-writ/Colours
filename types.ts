
export type LearningState = 
  | 'flow' 
  | 'neutral' 
  | 'stress' 
  | 'fatigue' 
  | 'distraction' 
  | 'loss_of_interest' 
  | 'low_learning';

export interface StudyData {
  heart_rate: number;
  hrv: number;
  state: LearningState;
  learning_capacity: number;
  message: string;
  topic: string;
}

export interface HistoryPoint {
  time: string;
  heart_rate: number;
  hrv: number;
  learning_capacity: number;
}

export interface LearningSuggestion {
  title: string;
  content: string;
  type: 'summary' | 'example' | 'simpler' | 'key_concepts';
}
