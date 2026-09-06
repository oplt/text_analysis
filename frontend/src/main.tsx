import React from "react";
import ReactDOM from "react-dom/client";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "./index.css";
import { AppProviders } from "./app/providers";
import { AppRouter } from "./app/router";

ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
        <AppProviders>
            <AppRouter />
        </AppProviders>
    </React.StrictMode>
);
