import React, { useState } from "react";
import { useSelector } from "react-redux";
import Sidebar from "./components/common/Sidebar";
import Topbar from "./components/common/Topbar";
import AppRouter from "./routers/AppRouter";
import ErrorBoundary from "./components/common/ErrorBoundary";
import { AddPatientModal } from "./components/common/AddPatientModal";
import type { RootState } from "./stores/store";

const App: React.FC = () => {
  const isAuthenticated = useSelector((state: RootState) => state.auth.isAuthenticated);
  const [isAddPatientModalOpen, setIsAddPatientModalOpen] = useState(false);

  React.useEffect(() => {
    const handleOpen = () => setIsAddPatientModalOpen(true);
    window.addEventListener("open-add-patient-modal", handleOpen);
    return () => window.removeEventListener("open-add-patient-modal", handleOpen);
  }, []);

  return (
    <ErrorBoundary>
      {isAuthenticated ? (
        <div className="flex h-screen w-screen overflow-hidden bg-background text-on-background dark:bg-[#1a1d27] dark:text-gray-100 font-body-md text-body-md antialiased">
          {/* Sidebar Navigation */}
          <Sidebar onNewAnalysisClick={() => setIsAddPatientModalOpen(true)} />

          {/* Main Workspace Area */}
          <div className="flex-1 flex flex-col md:pl-64 h-full relative overflow-hidden">
            {/* Top Header Bar */}
            <Topbar />

            {/* Page Routing Container */}
            <main className="flex-1 overflow-y-auto bg-surface-container-lowest dark:bg-[#1a1d27] relative">
              <AppRouter />
            </main>
          </div>

          {/* Add Patient & Scan X-Ray Modal */}
          <AddPatientModal
            isOpen={isAddPatientModalOpen}
            onClose={() => setIsAddPatientModalOpen(false)}
          />
        </div>
      ) : (
        <div className="h-screen w-screen overflow-hidden bg-[#f8fafc] dark:bg-[#151722]">
          <AppRouter />
        </div>
      )}
    </ErrorBoundary>
  );
};

export default App;


