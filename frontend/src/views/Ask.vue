<script setup lang="ts">
// Chat screen: thread of questions and answers for the selected knowledge base, with the composer pinned below.
// The thread itself lives in chat.ts so the sidebar can reset it or reopen a past question.
import { computed, nextTick, onMounted, ref, watch } from "vue";
import { RouterLink, useRoute } from "vue-router";
import { api, describeError } from "../api.ts";
import { ask, chat, selectKb } from "../chat.ts";
import { MAX_QUESTION_CHARS, suggestionGroups } from "../lib.ts";
import { loadKbs, session } from "../session.ts";
import type { Doc } from "../types.ts";
import ChatMessage from "../components/ChatMessage.vue";
import Composer from "../components/Composer.vue";
import SuggestionCards from "../components/SuggestionCards.vue";
import glow from "../assets/glow.svg";
import logo from "../assets/logo.png";

const route = useRoute();
const loading = ref(true);
const error = ref("");
const composer = ref<InstanceType<typeof Composer>>();
const thread = ref<HTMLElement>();

const kbs = computed(() => session.kbs ?? []);
const empty = computed(() => !chat.turns.length);

// Documents of the selected KB feed the empty-state cards.
const docs = ref<Doc[]>([]);
const docsState = ref<"loading" | "ok" | "error">("loading");
const docsError = ref("");
const groups = computed(() => suggestionGroups(docs.value));
let docSeq = 0;
watch(
  () => chat.kbId,
  async (id) => {
    docs.value = [];
    const mine = ++docSeq; // also drops a late answer for the previous KB when none is selected now
    if (!id) return; // the template shows the "no active knowledge base" note instead
    docsState.value = "loading";
    try {
      // ponytail: one page of up to 1000 documents; the cards only need a sample.
      const r = await api<Doc[]>(`/kbs/${id}/documents?status=ready&limit=1000`);
      if (mine !== docSeq) return;
      docs.value = r;
      docsState.value = "ok";
    } catch (e) {
      if (mine !== docSeq) return;
      docsError.value = describeError(e);
      docsState.value = "error";
    }
  },
  { immediate: true },
);

function focusIfWanted() {
  if (!chat.wantFocus) return;
  chat.wantFocus = false;
  void nextTick(() => composer.value?.focus());
}
watch(() => chat.wantFocus, focusIfWanted);

onMounted(async () => {
  try {
    await loadKbs();
    const wanted = typeof route.query.kb === "string" ? route.query.kb : "";
    const cur = kbs.value.find((k) => k.id === chat.kbId);
    const keep = cur && (cur.status === "active" || chat.turns.length > 0); // a reopened answer may belong to an archived KB
    const pick = kbs.value.find((k) => k.id === wanted && k.status === "active") ?? (keep ? cur : undefined) ?? kbs.value.find((k) => k.status === "active" && k.doc_count > 0) ?? kbs.value.find((k) => k.status === "active");
    selectKb(pick?.id ?? "");
  } catch (e) {
    error.value = describeError(e);
  } finally {
    loading.value = false;
  }
  focusIfWanted();
});

// Show the newest question at the top of the thread whenever a turn is added or its answer arrives.
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
watch(
  () => [chat.turns.length, chat.turns[chat.turns.length - 1]?.status],
  async () => {
    await nextTick();
    const last = chat.turns[chat.turns.length - 1];
    thread.value?.querySelector(`[data-turn="${last?.id}"]`)?.scrollIntoView({ block: "start", behavior: reduceMotion.matches ? "auto" : "smooth" });
  },
);

function prefill(title: string) {
  chat.draft = `Apa saja pokok pengaturan dalam ${title}?`.slice(0, MAX_QUESTION_CHARS);
  composer.value?.focus();
}
</script>

<template>
  <section class="chat" aria-labelledby="h-ask">
    <h1 id="h-ask" class="sr-only">Tanya</h1>
    <img v-if="empty && !loading" class="glow" :src="glow" alt="" aria-hidden="true" />

    <p v-if="loading" class="chat-note" role="status">Memuat…</p>
    <p v-else-if="error" class="alert bad chat-note" role="alert">{{ error }}</p>
    <p v-else-if="!kbs.length" class="chat-note">
      Kamu belum punya akses ke knowledge base mana pun. <RouterLink :to="{ name: 'kbs' }">Lihat daftar</RouterLink>
    </p>
    <p v-else-if="!chat.kbId" class="chat-note">
      Tidak ada knowledge base aktif untuk ditanya: semua knowledge base yang bisa kamu akses sedang diarsipkan. <RouterLink :to="{ name: 'kbs' }">Lihat daftar</RouterLink>
    </p>

    <template v-else>
      <div ref="thread" class="thread">
        <div v-if="empty" class="empty">
          <img :src="logo" alt="" width="59" height="67" />
          <p v-if="docsState === 'loading'" class="muted" role="status">Memuat dokumen…</p>
          <p v-else-if="docsState === 'error'" class="muted">{{ docsError }}</p>
          <p v-else-if="!groups.length" class="muted empty-msg">Belum ada dokumen yang siap dicari di knowledge base ini. Unggah dokumen lewat halaman Knowledge Base, lalu tunggu proses indexnya selesai.</p>
          <SuggestionCards v-else :groups="groups" @pick="prefill" />
        </div>
        <div v-else class="turns" role="log" aria-live="polite" aria-relevant="additions">
          <ChatMessage v-for="t in chat.turns" :key="t.id" :turn="t" />
        </div>
      </div>

      <div class="composer-wrap">
        <Composer ref="composer" v-model="chat.draft" :kbs="kbs" :kb-id="chat.kbId" :busy="chat.busy" :max-length="MAX_QUESTION_CHARS" @update:kbId="selectKb" @submit="ask" />
      </div>
    </template>
  </section>
</template>
