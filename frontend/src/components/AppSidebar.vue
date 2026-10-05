<script setup lang="ts">
// Sidebar of every signed-in page: brand, "Pertanyaan baru", page links, question history and the profile card.
import { computed, nextTick, ref } from "vue";
import { RouterLink, useRoute, useRouter } from "vue-router";
import { chat, newThread, removeHistory, restore } from "../chat.ts";
import { initial } from "../lib.ts";
import { canEdit, roleLabel, session } from "../session.ts";
import type { HistoryEntry } from "../types.ts";
import DropdownMenu from "./DropdownMenu.vue";
import Icon from "./Icon.vue";
import logo from "../assets/logo.png";
import fileIcon from "../assets/file.svg";
import logoutIcon from "../assets/logout.svg";
import moreIcon from "../assets/more.svg";
import navAsk from "../assets/nav-ask.svg";
import navKb from "../assets/nav-kb.svg";
import navLog from "../assets/nav-log.svg";
import navUsers from "../assets/nav-users.svg";
import plusIcon from "../assets/plus.svg";
import toggleIcon from "../assets/sidebar-toggle.svg";
import trashIcon from "../assets/trash.svg";

defineProps<{ open: boolean }>();
const emit = defineEmits<{ close: []; navigate: []; logout: [] }>();

const route = useRoute();
const router = useRouter();
const toggleBtn = ref<HTMLButtonElement>();
const historyHeading = ref<HTMLElement>();
const historyList = ref<HTMLElement>();

// Logs are for admins and editors of at least one KB.
const showLogs = computed(() => !!session.me?.is_admin || !!session.kbs?.some(canEdit));
// Hide entries of knowledge bases the user can no longer open.
const history = computed(() => chat.history.filter((e) => !session.kbs || session.kbs.some((k) => k.id === e.kbId)));

async function goAsk() {
  if (route.name !== "ask") await router.push({ name: "ask" });
  emit("navigate");
}
async function onNew() {
  newThread();
  await goAsk();
}
async function onRestore(e: HistoryEntry) {
  restore(e);
  await goAsk();
}
// The focused row disappears: focus the entry that took its place (or the new last one), else the heading.
async function onRemove(id: string) {
  const i = history.value.findIndex((e) => e.id === id);
  removeHistory(id);
  await nextTick();
  const rows = historyList.value?.querySelectorAll<HTMLElement>(".hist-main");
  (rows?.[Math.min(i, rows.length - 1)] ?? historyHeading.value)?.focus();
}

defineExpose({ focusToggle: () => toggleBtn.value?.focus() });
</script>

<template>
  <aside id="sidebar" class="sidebar" :class="{ 'is-open': open }" aria-label="Menu samping">
    <div class="side-head">
      <RouterLink class="brand" :to="{ name: 'ask' }" @click="emit('navigate')">
        <img :src="logo" alt="" width="33" height="38" />
        <span>IDEAS Ask</span>
      </RouterLink>
      <button ref="toggleBtn" type="button" class="icon-btn" aria-label="Tutup sidebar" @click="emit('close')">
        <Icon :src="toggleIcon" :size="22" />
      </button>
    </div>

    <button type="button" class="side-new" @click="onNew">
      <Icon :src="plusIcon" :size="24" />
      <span>Pertanyaan baru</span>
    </button>

    <nav class="side-nav" aria-label="Navigasi utama">
      <RouterLink :to="{ name: 'ask' }" @click="emit('navigate')"><Icon :src="navAsk" :size="18" /> Tanya</RouterLink>
      <RouterLink :to="{ name: 'kbs' }" @click="emit('navigate')"><Icon :src="navKb" :size="18" /> Knowledge Base</RouterLink>
      <RouterLink v-if="showLogs" :to="{ name: 'logs' }" @click="emit('navigate')"><Icon :src="navLog" :size="18" /> Log</RouterLink>
      <RouterLink v-if="session.me?.is_admin" :to="{ name: 'users' }" @click="emit('navigate')"><Icon :src="navUsers" :size="18" /> Pengguna</RouterLink>
    </nav>

    <section class="side-history" aria-labelledby="h-history">
      <h2 id="h-history" ref="historyHeading" tabindex="-1">Riwayat</h2>
      <p v-if="!history.length" class="muted small">Pertanyaanmu akan muncul di sini.</p>
      <ul v-else ref="historyList">
        <li v-for="e in history" :key="e.id" class="hist">
          <button type="button" class="hist-main" :title="e.kbName ? `${e.question} (${e.kbName})` : e.question" @click="onRestore(e)">
            <Icon :src="fileIcon" :size="18" />
            <span>{{ e.question }}</span>
          </button>
          <DropdownMenu :label="`Aksi untuk: ${e.question}`" align="end">
            <template #trigger="{ attrs, toggle }">
              <button type="button" class="icon-btn hist-more" v-bind="attrs" :aria-label="`Menu untuk: ${e.question}`" @click="toggle">
                <Icon :src="moreIcon" :size="20" />
              </button>
            </template>
            <template #default="{ close }">
              <button type="button" role="menuitem" class="menu-item" @click="close(false); onRemove(e.id)">
                <Icon :src="trashIcon" :size="20" /> Hapus
              </button>
            </template>
          </DropdownMenu>
        </li>
      </ul>
    </section>

    <div v-if="session.me" class="profile-card">
      <DropdownMenu label="Menu akun" placement="top" match-width>
        <template #trigger="{ attrs, toggle }">
          <button type="button" class="profile" v-bind="attrs" @click="toggle">
            <span class="avatar" aria-hidden="true">{{ initial(session.me!.name) }}</span>
            <span class="profile-text">
              <span class="profile-name">{{ session.me!.name }}</span>
              <span class="profile-mail">{{ session.me!.email }}</span>
            </span>
            <span class="role-badge">{{ roleLabel() }}</span>
          </button>
        </template>
        <template #default="{ close }">
          <button type="button" role="menuitem" class="menu-item" @click="close(false); emit('logout')">
            <Icon :src="logoutIcon" :size="18" /> Keluar
          </button>
        </template>
      </DropdownMenu>
    </div>
  </aside>
</template>
