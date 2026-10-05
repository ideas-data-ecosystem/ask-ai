import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  base: "/",
  plugins: [vue()],
  server: {
    // Keep the browser's Host header: the API's CSRF check compares Origin with Host.
    proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: false } },
  },
});
