<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { useRoute } from "vue-router";
import { api, describeError } from "../api.ts";
import { fmtDate } from "../lib.ts";
import { canEdit, loadKbs, session } from "../session.ts";
import type { Job, QueryLog } from "../types.ts";
import StatusBadge from "../components/StatusBadge.vue";

const LIMIT = 100;
const route = useRoute();
const kbId = ref("");
const tab = ref<"jobs" | "queries">("jobs");
const jobStatus = ref("");
const onlyInsufficient = ref(false);
const jobs = ref<Job[]>([]);
const queries = ref<QueryLog[]>([]);
const more = ref(false);
const loading = ref(true);
const error = ref("");

const kbs = computed(() => (session.kbs ?? []).filter(canEdit)); // logs are editor+ per the contract
let seq = 0;

async function load(append = false) {
  const mine = ++seq; // ignore responses from a superseded request
  loading.value = true;
  error.value = "";
  const q = new URLSearchParams({ limit: String(LIMIT), offset: String(append ? (tab.value === "jobs" ? jobs.value.length : queries.value.length) : 0) });
  if (tab.value === "jobs" && jobStatus.value) q.set("status", jobStatus.value);
  if (tab.value === "queries" && onlyInsufficient.value) q.set("insufficient", "true");
  try {
    if (tab.value === "jobs") {
      const rows = await api<Job[]>(`/kbs/${kbId.value}/jobs?${q}`);
      if (mine !== seq) return;
      jobs.value = append ? [...jobs.value, ...rows] : rows;
      more.value = rows.length === LIMIT;
    } else {
      const rows = await api<QueryLog[]>(`/kbs/${kbId.value}/queries?${q}`);
      if (mine !== seq) return;
      queries.value = append ? [...queries.value, ...rows] : rows;
      more.value = rows.length === LIMIT;
    }
  } catch (e) {
    if (mine === seq) error.value = describeError(e);
  } finally {
    if (mine === seq) loading.value = false;
  }
}

onMounted(async () => {
  try {
    await loadKbs();
  } catch (e) {
    error.value = describeError(e);
  }
  const wanted = typeof route.query.kb === "string" ? route.query.kb : "";
  kbId.value = (kbs.value.find((k) => k.id === wanted) ?? kbs.value[0])?.id ?? "";
  loading.value = false;
});

watch(kbId, () => {
  jobs.value = [];
  queries.value = [];
});
watch([kbId, tab, jobStatus, onlyInsufficient], () => {
  if (kbId.value) void load();
});

const tokens = (q: QueryLog) => (q.prompt_tokens ?? 0) + (q.completion_tokens ?? 0) || "-";
</script>

<template>
  <h1>Log</h1>
  <p v-if="error" class="alert bad" role="alert">{{ error }}</p>
  <p v-if="loading && !kbs.length" class="muted" role="status">Memuat…</p>
  <p v-else-if="!kbs.length && !error" class="muted">Log hanya tersedia untuk admin dan editor knowledge base.</p>

  <template v-if="kbs.length">
    <div class="card filters">
      <div class="field">
        <label for="l-kb">Knowledge base</label>
        <select id="l-kb" v-model="kbId">
          <option v-for="k in kbs" :key="k.id" :value="k.id">{{ k.name }}</option>
        </select>
      </div>
      <div class="field">
        <span class="label">Jenis log</span>
        <div class="row">
          <button type="button" class="btn sm" :class="{ primary: tab === 'jobs' }" :aria-pressed="tab === 'jobs'" @click="tab = 'jobs'">Job index</button>
          <button type="button" class="btn sm" :class="{ primary: tab === 'queries' }" :aria-pressed="tab === 'queries'" @click="tab = 'queries'">Kueri</button>
        </div>
      </div>
      <div v-if="tab === 'jobs'" class="field">
        <label for="l-status">Status</label>
        <select id="l-status" v-model="jobStatus">
          <option value="">Semua</option>
          <option value="queued">Antre</option>
          <option value="running">Berjalan</option>
          <option value="done">Selesai</option>
          <option value="failed">Gagal</option>
        </select>
      </div>
      <label v-else class="check"><input v-model="onlyInsufficient" type="checkbox" /> Hanya yang tidak cukup informasi</label>
      <button type="button" class="btn sm" :disabled="loading" @click="load()">Muat ulang</button>
    </div>

    <p v-if="loading" class="muted" role="status">Memuat…</p>

    <template v-if="tab === 'jobs'">
      <p v-if="!loading && !jobs.length && !error" class="muted">Belum ada job.</p>
      <table v-if="jobs.length" class="table">
        <thead>
          <tr>
            <th scope="col">#</th><th scope="col">Berkas</th><th scope="col">Status</th><th scope="col">Percobaan</th>
            <th scope="col">Hasil</th><th scope="col">Dibuat</th><th scope="col">Selesai</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="j in jobs" :key="j.id">
            <td data-label="#">{{ j.id }}</td>
            <td data-label="Berkas">
              <div>
                {{ j.filename }}
                <div v-if="j.error" class="error-text">Galat: {{ j.error }}</div>
              </div>
            </td>
            <td data-label="Status"><StatusBadge :status="j.status" /></td>
            <td data-label="Percobaan">{{ j.attempts }}</td>
            <td data-label="Hasil">
              <span v-if="j.status === 'done'">{{ j.stats.pages ?? 0 }} hlm, {{ j.stats.ocr_pages ?? 0 }} OCR, {{ j.stats.chunks ?? 0 }} chunk, {{ j.stats.seconds ?? 0 }} dtk</span>
              <span v-else>-</span>
            </td>
            <td data-label="Dibuat">{{ fmtDate(j.created_at) }}</td>
            <td data-label="Selesai">{{ fmtDate(j.finished_at) }}</td>
          </tr>
        </tbody>
      </table>
    </template>

    <template v-else>
      <p v-if="!loading && !queries.length && !error" class="muted">Belum ada kueri.</p>
      <table v-if="queries.length" class="table">
        <thead>
          <tr>
            <th scope="col">Waktu</th><th scope="col">Pengguna</th><th scope="col">Pertanyaan</th><th scope="col">Hasil</th>
            <th scope="col">Latensi</th><th scope="col">Token</th><th scope="col">Rincian</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="q in queries" :key="q.id">
            <td data-label="Waktu">{{ fmtDate(q.created_at) }}</td>
            <td data-label="Pengguna">{{ q.user_email ?? "-" }}</td>
            <td data-label="Pertanyaan">
              <div>
                {{ q.question }}
                <div v-if="q.history_turns" class="muted small">Lanjutan dari {{ q.history_turns }} giliran sebelumnya</div>
                <div v-if="q.rewritten_question" class="muted small">Dicari sebagai: {{ q.rewritten_question }}</div>
              </div>
            </td>
            <td data-label="Hasil">
              <span v-if="q.error" class="badge bad"><span aria-hidden="true">✕</span> Galat</span>
              <span v-else-if="q.insufficient" class="badge warn"><span aria-hidden="true">ⓘ</span> Tidak cukup</span>
              <span v-else class="badge ok"><span aria-hidden="true">✓</span> Dijawab</span>
            </td>
            <td data-label="Latensi">{{ q.latency_ms != null ? `${q.latency_ms} ms` : "-" }}</td>
            <td data-label="Token">{{ tokens(q) }}</td>
            <td data-label="Rincian">
              <details>
                <summary>Lihat</summary>
                <p v-if="q.error" class="error-text">Galat: {{ q.error }}</p>
                <p class="pre">{{ q.answer ?? "(tanpa jawaban)" }}</p>
                <p class="muted small">Model: {{ q.model ?? "-" }}; {{ q.retrieved.length }} potongan dokumen diambil.</p>
              </details>
            </td>
          </tr>
        </tbody>
      </table>
    </template>

    <button v-if="more && !loading" type="button" class="btn" @click="load(true)">Muat lebih banyak</button>
  </template>
</template>
