import { createApp } from "vue";
import App from "./App.vue";
import router from "./router.ts";
import { hooks } from "./api.ts";
import { session } from "./session.ts";
import "@fontsource-variable/inter";
import "./style.css";

// A 401 anywhere means the session is gone: drop local state and go to the login page.
hooks.onUnauthorized = () => {
  session.me = null;
  session.kbs = null;
  if (router.currentRoute.value.name !== "login") {
    void router.replace({ name: "login", query: { redirect: router.currentRoute.value.fullPath } });
  }
};

createApp(App).use(router).mount("#app");
