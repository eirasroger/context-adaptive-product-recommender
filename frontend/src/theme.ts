const STORAGE_KEY = "theme";

type Theme = "light" | "dark";

export function restoreTheme() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) document.documentElement.setAttribute("data-theme", saved);
  } catch {}
}

export function toggleTheme() {
  const root = document.documentElement;
  const now = root.getAttribute("data-theme");
  const prefersDark = matchMedia("(prefers-color-scheme: dark)").matches;
  const next: Theme =
    now === "dark" ? "light" : now === "light" ? "dark" : prefersDark ? "light" : "dark";
  root.setAttribute("data-theme", next);
  try {
    localStorage.setItem(STORAGE_KEY, next);
  } catch {}
}
