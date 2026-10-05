import { createRouter, createWebHistory } from "vue-router";
import { loadMe, session } from "./session.ts";

declare module "vue-router" {
  interface RouteMeta {
    title: string;
    public?: boolean;
    admin?: boolean;
    full?: boolean; // page manages its own scrolling (the chat) instead of sitting in .page
  }
}

const router = createRouter({
  history: createWebHistory("/"),
  routes: [
    { path: "/login", name: "login", component: () => import("./views/Login.vue"), meta: { title: "Masuk", public: true } },
    { path: "/", redirect: { name: "kbs" } },
    { path: "/kbs", name: "kbs", component: () => import("./views/KbList.vue"), meta: { title: "Knowledge Base" } },
    { path: "/kbs/:id", name: "kb", component: () => import("./views/KbDetail.vue"), props: true, meta: { title: "Detail Knowledge Base" } },
    { path: "/ask", name: "ask", component: () => import("./views/Ask.vue"), meta: { title: "Tanya", full: true } },
    { path: "/logs", name: "logs", component: () => import("./views/Logs.vue"), meta: { title: "Log" } },
    { path: "/users", name: "users", component: () => import("./views/Users.vue"), meta: { title: "Pengguna", admin: true } },
    { path: "/:rest(.*)*", redirect: { name: "kbs" } },
  ],
});

router.beforeEach(async (to) => {
  if (!session.loaded) await loadMe();
  if (to.meta.public) return session.me ? { name: "kbs" } : true;
  if (!session.me) return { name: "login", query: { redirect: to.fullPath } };
  if (to.meta.admin && !session.me.is_admin) return { name: "kbs" };
  return true;
});

router.afterEach((to) => {
  document.title = `${to.meta.title} - IDEAS Ask`;
});

export default router;
