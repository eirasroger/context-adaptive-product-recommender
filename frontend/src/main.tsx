import { Analytics } from "@vercel/analytics/react";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { withoutFragment } from "./analytics";
import { App } from "./App";
import "@fontsource-variable/inter";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
    <Analytics beforeSend={withoutFragment} />
  </StrictMode>,
);
