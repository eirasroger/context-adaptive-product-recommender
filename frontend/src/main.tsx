import { Analytics } from "@vercel/analytics/react";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { withoutFragment } from "./analytics";
import { App } from "./App";
import "./styles.css";
import { restoreTheme } from "./theme";

restoreTheme();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
    <Analytics beforeSend={withoutFragment} />
  </StrictMode>,
);
