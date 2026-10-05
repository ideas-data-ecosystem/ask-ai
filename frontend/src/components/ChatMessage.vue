<script setup lang="ts">
// One question and its answer: user bubble, then the assistant side (typing placeholder, error, "not enough
// information" notice, or the answer with [n] chips, sources and a Copy action).
import { computed, nextTick, onBeforeUnmount, ref } from "vue";
import { copyText, fileUrl, initial, pageRange, splitAnswer } from "../lib.ts";
import { session } from "../session.ts";
import type { Turn } from "../types.ts";
import Icon from "./Icon.vue";
import logo from "../assets/logo.png";
import checkIcon from "../assets/check.svg";
import copyIcon from "../assets/copy.svg";
import infoIcon from "../assets/info.svg";

const props = defineProps<{ turn: Turn }>();

const active = ref<number | null>(null);
const copyState = ref<"" | "ok" | "fail">("");
let timer: ReturnType<typeof setTimeout> | undefined;
onBeforeUnmount(() => clearTimeout(timer));

const parts = computed(() => splitAnswer(props.turn.answer, new Set(props.turn.citations.map((c) => c.n))));
const sourceId = (n: number) => `src-${props.turn.id}-${n}`;

async function pick(n: number) {
  active.value = n;
  await nextTick();
  document.getElementById(sourceId(n))?.focus(); // scrolls into view and moves keyboard focus
}

async function copy() {
  copyState.value = (await copyText(props.turn.answer)) ? "ok" : "fail";
  clearTimeout(timer);
  timer = setTimeout(() => (copyState.value = ""), 2000);
}
</script>

<template>
  <article class="turn" :data-turn="turn.id">
    <div class="q-row">
      <h3 class="sr-only">Pertanyaan</h3>
      <p class="bubble">{{ turn.question }}</p>
      <span class="avatar" aria-hidden="true">{{ initial(session.me?.name ?? "") }}</span>
    </div>

    <div class="a-row">
      <span class="avatar bot" aria-hidden="true"><img :src="logo" alt="" width="17" height="20" /></span>
      <div class="a-body">
        <h3 class="sr-only">Jawaban</h3>

        <!-- No role="status"/"alert" in here: the thread is a role="log" live region and already announces
             each added node, so nested live regions would be read twice. -->
        <p v-if="turn.status === 'pending'" class="typing">
          <span class="dots" aria-hidden="true"><i></i><i></i><i></i></span>
          Mencari jawaban di dokumen…
        </p>

        <p v-else-if="turn.status === 'error'" class="alert bad">{{ turn.error }}</p>

        <div v-else-if="turn.insufficient" class="notice">
          <Icon :src="infoIcon" :size="20" />
          <div>
            <strong>Tidak cukup informasi di knowledge base ini</strong>
            <p>Dokumen di knowledge base ini belum memuat jawaban untuk pertanyaan tersebut. Coba ubah pertanyaan atau pilih knowledge base lain.</p>
          </div>
        </div>

        <template v-else>
          <p class="answer">
            <template v-for="(p, i) in parts" :key="i">
              <button v-if="'n' in p" type="button" class="chip" :class="{ on: active === p.n }" :aria-pressed="active === p.n" :aria-label="`Sumber ${p.n}`" @click="pick(p.n)">{{ p.n }}</button>
              <template v-else>{{ p.text }}</template>
            </template>
          </p>

          <ol v-if="turn.citations.length" class="sources" aria-label="Sumber">
            <li v-for="c in turn.citations" :id="sourceId(c.n)" :key="c.n" tabindex="-1" class="source" :class="{ on: active === c.n }" :aria-current="active === c.n ? 'true' : undefined">
              <div class="source-top">
                <span class="chip static">{{ c.n }}</span>
                <strong class="source-name">{{ c.filename }}</strong>
              </div>
              <p class="source-meta">
                <span v-if="c.category">{{ c.category }}</span>
                <span v-if="c.page_start != null">Hlm. {{ pageRange(c) }}</span>
              </p>
              <p v-if="c.heading" class="source-heading">{{ c.heading }}</p>
              <p class="snippet">{{ c.snippet }}</p>
              <a class="source-link" :href="fileUrl(turn.kbId, c.document_id, c.page_start)" target="_blank" rel="noopener">
                {{ /\.pdf$/i.test(c.filename) ? "Buka PDF" : "Buka berkas" }}{{ c.page_start != null ? ` (hlm. ${c.page_start})` : "" }}
              </a>
            </li>
          </ol>

          <div class="actions">
            <button type="button" class="act" aria-label="Salin jawaban" @click="copy">
              <Icon :src="copyState === 'ok' ? checkIcon : copyIcon" :size="20" />
              <span v-if="copyState" class="act-note">{{ copyState === "ok" ? "Tersalin" : "Gagal menyalin" }}</span>
            </button>
          </div>
        </template>
      </div>
    </div>
  </article>
</template>
