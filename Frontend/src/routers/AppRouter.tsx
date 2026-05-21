import React, { lazy, Suspense } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { useSelector } from "react-redux";
import type { RootState } from "../stores/store";
import PageLoading from "../components/common/PageLoading";
import ProtectedRoute from "../components/common/ProtectedRoute";

const DashboardPage = lazy(() => import("../modules/dashboard/pages/DashboardPage"));
const PatientHistoryPage = lazy(() => import("../modules/history/pages/PatientHistoryPage"));
const PatientProfilePage = lazy(() => import("../modules/patients/pages/PatientProfilePage"));
const ResearchPortalPage = lazy(() => import("../modules/research/pages/ResearchPortalPage"));
const SystemConfigPage = lazy(() => import("../modules/settings/pages/SystemConfigPage"));
const LoginPage = lazy(() => import("../modules/auth/pages/LoginPage"));
const RegisterPage = lazy(() => import("../modules/auth/pages/RegisterPage"));

const AppRouter: React.FC = () => {
  const isAuthenticated = useSelector((state: RootState) => state.auth.isAuthenticated);

  return (
    <Suspense fallback={<PageLoading />}>
      <Routes>
        {/* Auth routes */}
        <Route
          path="/login"
          element={isAuthenticated ? <Navigate to="/" replace /> : <LoginPage />}
        />
        <Route
          path="/register"
          element={isAuthenticated ? <Navigate to="/" replace /> : <RegisterPage />}
        />

        {/* Protected routes */}
        <Route
          path="/"
          element={
            <ProtectedRoute>
              <DashboardPage />
            </ProtectedRoute>
          }
        />
        <Route path="/diagnostic-hub" element={<Navigate to="/" replace />} />
        <Route
          path="/patient-history"
          element={
            <ProtectedRoute>
              <PatientHistoryPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/patient-profile/:id"
          element={
            <ProtectedRoute>
              <PatientProfilePage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/research-portal"
          element={
            <ProtectedRoute>
              <ResearchPortalPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/system-config"
          element={
            <ProtectedRoute>
              <SystemConfigPage />
            </ProtectedRoute>
          }
        />

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Suspense>
  );
};

export default AppRouter;

