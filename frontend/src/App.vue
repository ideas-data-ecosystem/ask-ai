<script setup lang="ts">
import { nextTick, onBeforeUnmount, ref, watch } from "vue";
import { RouterView, useRoute, useRouter } from "vue-router";
import AppSidebar from "./components/AppSidebar.vue";
import Icon from "./components/Icon.vue";
import { loadKbs, logout, session } from "./session.ts";
import logo from "./assets/logo.png";
import toggleIcon from "./assets/sidebar-toggle.svg";

const route = useRoute();
const router = useRouter();

// Desktop: the sidebar sits beside the page and can be collapsed. Below 768px it is an off-canvas drawer.
const mobileQuery = window.matchMedia("(max-width: 767px)");
const mobile = ref(mobileQuery.matches);
const sidebarOpen = ref(!mobile.value && window.innerWidth >= 1024);
const sidebar = ref<InstanceType<typeof AppSidebar>>();
const opener = ref<HTMLButtonElement>();
const main = ref<HTMLElement>();

const onBreakpoint = () => {
  mobile.value = mobileQuery.matches;
  sidebarOpen.value = !mobile.value && window.innerWidth >= 1024;
};
mobileQuery.addEventListener("change", onBreakpoint);
onBeforeUnmount(() => mobileQuery.removeEventListener("change", onBreakpoint));

async function openSidebar() {
  sidebarOpen.value = true;
  await nextTick();
  sidebar.value?.focusToggle();
}
async function closeSidebar() {
  sidebarOpen.value = false;
  await nextTick();
  opener.value?.focus(); // focus returns to the button that reopens it
}
// Called after any sidebar action: on a phone the drawer gets out of the way and focus moves to the page,
// unless the page already took it (the composer after "Pertanyaan baru").
async function onNavigate() {
  if (!mobile.value) return;
  sidebarOpen.value = false;
  await nextTick(); // .shell-main is no longer inert
  const a = document.activeElement;
  if (!a || a === document.body || a.closest("#sidebar")) main.value?.focus();
}
function onKeydown(e: KeyboardEvent) {
  if (e.key === "Escape" && mobile.value && sidebarOpen.value && !e.defaultPrevented) void closeSidebar();
}
window.addEventListener("keydown", onKeydown);
onBeforeUnmount(() => window.removeEventListener("keydown", onKeydown));

watch(
  () => session.me?.id,
  (id) => {
    if (id) loadKbs().catch(() => {}); // pages show their own errors; this only feeds the sidebar
  },
  { immediate: true },
);

async function onLogout() {
  await logout().catch(() => {});
  await router.replace({ name: "login" });
}
</script>

<template>
  <a class="skip-link" href="#main">Lewati ke konten</a>
  <div v-if="session.me && !route.meta.public" class="app" :class="{ 'is-collapsed': !sidebarOpen }">
    <AppSidebar ref="sidebar" :open="sidebarOpen" @close="closeSidebar" @navigate="onNavigate" @logout="onLogout" />
    <div v-if="mobile && sidebarOpen" class="backdrop" aria-hidden="true" @click="closeSidebar"></div>
    <div class="shell-main" :inert="mobile && sidebarOpen">
      <header class="topbar">
        <button ref="opener" type="button" class="icon-btn" aria-label="Buka sidebar" aria-controls="sidebar" :aria-expanded="sidebarOpen" @click="openSidebar">
          <Icon :src="toggleIcon" :size="22" />
        </button>
        <img :src="logo" alt="" width="26" height="30" />
        <span class="topbar-name">IDEAS Ask</span>
      </header>
      <main id="main" ref="main" tabindex="-1" class="content">
        <RouterView v-if="route.meta.full" :key="route.path" />
        <div v-else class="page"><RouterView :key="route.path" /></div>
      </main>
    </div>
  </div>
  <RouterView v-else-if="route.meta.public" />
</template>
