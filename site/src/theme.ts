/** Light/dark theme: a stored choice wins, otherwise the system preference. */

const KEY = "broadsheet-theme";

export type Theme = "light" | "dark";

type Listener = (theme: Theme) => void;
const listeners = new Set<Listener>();

function systemTheme(): Theme {
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

function stored(): Theme | null {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" ? v : null;
  } catch {
    return null;
  }
}

export function currentTheme(): Theme {
  const attr = document.documentElement.getAttribute("data-theme");
  if (attr === "light" || attr === "dark") return attr;
  return systemTheme();
}

function apply(theme: Theme): void {
  document.documentElement.setAttribute("data-theme", theme);
  const btn = document.getElementById("theme-toggle");
  if (btn) {
    btn.setAttribute("aria-pressed", String(theme === "dark"));
    btn.setAttribute(
      "aria-label",
      theme === "dark"
        ? "Dark theme, switch to light"
        : "Light theme, switch to dark",
    );
  }
  for (const cb of listeners) cb(theme);
}

/**
 * Adopt the theme the inline head script already painted, and follow the
 * system preference for as long as the reader has not chosen one.
 */
export function initTheme(): void {
  apply(stored() ?? systemTheme());
  window
    .matchMedia("(prefers-color-scheme: dark)")
    .addEventListener("change", (e) => {
      if (stored() === null) apply(e.matches ? "dark" : "light");
    });
}

export function toggleTheme(): void {
  const next: Theme = currentTheme() === "dark" ? "light" : "dark";
  try {
    localStorage.setItem(KEY, next);
  } catch {
    /* private mode: the choice just does not persist */
  }
  apply(next);
}

/** Charts re-render on theme change; they read their colours from CSS vars. */
export function onThemeChange(cb: Listener): void {
  listeners.add(cb);
}

/** Read a CSS custom property off :root, e.g. cssVar("--s1"). */
export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();
}
