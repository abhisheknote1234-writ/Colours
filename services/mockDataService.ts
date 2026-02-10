
import { StudyData, LearningState } from '../types';
import { MOCK_TOPICS } from '../constants';

let currentHR = 72;
let currentHRV = 45;
let currentCapacity = 85;
let tick = 0;

const STATES: LearningState[] = ['flow', 'neutral', 'stress', 'fatigue', 'distraction', 'loss_of_interest', 'low_learning'];

export const fetchRealtimeData = async (): Promise<StudyData> => {
  // Simulate network delay
  await new Promise(resolve => setTimeout(resolve, 300));

  tick++;
  
  // Random walk for data to look "real"
  currentHR += (Math.random() - 0.5) * 4;
  currentHR = Math.max(60, Math.min(110, currentHR));

  currentHRV += (Math.random() - 0.5) * 6;
  currentHRV = Math.max(10, Math.min(90, currentHRV));

  currentCapacity += (Math.random() - 0.5) * 5;
  currentCapacity = Math.max(0, Math.min(100, currentCapacity));

  // Logic based state determination (simple rules)
  let state: LearningState = 'neutral';
  if (currentCapacity > 80 && currentHR < 85) state = 'flow';
  else if (currentHR > 95) state = 'stress';
  else if (currentHRV < 25) state = 'fatigue';
  else if (currentCapacity < 40) state = 'low_learning';
  
  // Occasionally switch state randomly for demo purposes
  if (tick % 15 === 0) {
    state = STATES[Math.floor(Math.random() * STATES.length)];
  }

  const messages: Record<LearningState, string> = {
    flow: "This is a great window for complex problem solving.",
    neutral: "You're making steady progress. Keep going.",
    stress: "Breathing slowly for a minute might help clear the fog.",
    fatigue: "Maybe stand up and stretch? Your brain needs oxygen.",
    distraction: "Try to find one key point to re-engage with the topic.",
    loss_of_interest: "Would looking at a real-world example make this more interesting?",
    low_learning: "The current pace is tough. Let's try a different approach."
  };

  return {
    heart_rate: Math.round(currentHR),
    hrv: Math.round(currentHRV),
    state,
    learning_capacity: Math.round(currentCapacity),
    message: messages[state],
    topic: MOCK_TOPICS[Math.floor(tick / 50) % MOCK_TOPICS.length]
  };
};
