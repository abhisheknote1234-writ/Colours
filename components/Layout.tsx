
import React from 'react';

interface LayoutProps {
  children: React.ReactNode;
  connected: boolean;
}

const Layout: React.FC<LayoutProps> = ({ children, connected }) => {
  return (
    <div className="min-h-screen flex flex-col max-w-5xl mx-auto p-4 md:p-6 lg:p-8">
      <header className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
            Live Study Monitoring
          </h1>
          <p className="text-slate-500 text-sm italic">“Don’t force learning. Understand first.”</p>
        </div>
        
        <div className="flex items-center gap-2 bg-white px-4 py-2 rounded-full shadow-sm border border-slate-100">
          <div className={`w-2 h-2 rounded-full ${connected ? 'bg-emerald-500 animate-pulse' : 'bg-slate-300'}`} />
          <span className="text-xs font-medium text-slate-600 uppercase tracking-wider">
            {connected ? 'Connected' : 'Searching for Sensor...'}
          </span>
        </div>
      </header>
      
      <main className="flex-1 flex flex-col gap-6">
        {children}
      </main>
      
      <footer className="mt-12 py-6 border-t border-slate-200 text-center text-slate-400 text-xs">
        &copy; 2024 StudySense Labs • Cognitive Regulation Guidance
      </footer>
    </div>
  );
};

export default Layout;
