import { defineConfig } from "vite";

// GitHub Pages serves the site from /broadsheet/. VITE_BASE lets a local or
// alternative host override it without editing this file.
export default defineConfig({
  base: process.env.VITE_BASE ?? "/broadsheet/",
  build: {
    target: "es2020",
    chunkSizeWarningLimit: 2000,
    rollupOptions: {
      output: {
        // Vite 8 runs rolldown, which only accepts the function form.
        // Plotly is half a megabyte gzipped; keep it in one cacheable chunk.
        manualChunks(id) {
          if (id.includes("plotly.js-cartesian-dist-min")) return "plotly";
          return undefined;
        },
      },
    },
  },
});
