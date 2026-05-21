import React, { StrictMode, useEffect } from "react";
import { createRoot } from "react-dom/client";
import { Provider, useSelector } from "react-redux";
import { BrowserRouter } from "react-router-dom";
import { ConfigProvider, theme } from "antd";
import store from "./stores/store";
import type { RootState } from "./stores/store";
import App from "./App";
import "./index.css";

const AntDThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const themeMode = useSelector((state: RootState) => state.theme.mode);

  // Synchronize dynamic dark/light mode class list on document root element
  useEffect(() => {
    if (themeMode === "dark") {
      document.documentElement.classList.add("dark");
      document.documentElement.classList.remove("light");
    } else {
      document.documentElement.classList.remove("dark");
      document.documentElement.classList.add("light");
    }
  }, [themeMode]);

  return (
    <ConfigProvider
      theme={{
        algorithm: themeMode === "dark" ? theme.darkAlgorithm : theme.defaultAlgorithm,
        token: {
          colorPrimary: "#004ac6",
          borderRadius: 16,
          fontFamily: "Inter, sans-serif",
        },
      }}
    >
      {children}
    </ConfigProvider>
  );
};

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Provider store={store}>
      <BrowserRouter>
        <AntDThemeProvider>
          <App />
        </AntDThemeProvider>
      </BrowserRouter>
    </Provider>
  </StrictMode>
);
