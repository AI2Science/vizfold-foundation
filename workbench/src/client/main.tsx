import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App.tsx";
import "./app.css";

const host = document.getElementById("root");
if (!host) throw new Error("index.html is missing #root");

createRoot(host).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
