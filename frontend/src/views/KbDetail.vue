<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref } from "vue";
import { RouterLink, useRouter } from "vue-router";
import { api, ApiError, describeError } from "../api.ts";
import { fmtDate } from "../lib.ts";
import { canEdit, loadKbs } from "../session.ts";
import type { Doc, KB, KbStats } from "../types.ts";
import ConfirmButton from "../components/ConfirmButton.vue";
import DocumentsPanel from "../components/DocumentsPanel.vue";
import MembersPanel from "../components/MembersPanel.vue";
import StatusBadge from "../components/StatusBadge.vue";

const props = defineProps<{ id: string }>();
const router = useRouter();

const kb = ref<KB | null>(null);
const stats = ref<KbStats | null>(null);
const docs = ref<Doc[]>([]);
const loading = ref(true);
const error = ref(""); // load/poll error
const actionError = ref("");
const notice = ref("");
const busy = ref(false);
const form = reactive({ name: "", description: "", status: "active" as KB["status"] });

const isAdmin = computed(() => kb.value?.role === "admin");
const editor = computed(() => !!kb.value && canEdit(kb.value));
const active = computed(() => kb.value?.status === "active");

// Poll while any document is queued or processing (reindexing sets a document back to "queued").
const pending = computed(() => docs.value.some((d) => d.status === "queued" || d.status === "processing"));

let timer: ReturnType<typeof setTimeout> | undefined;
let alive = true;
onBeforeUnmount(() => {
  alive = false;
  clearTimeout(timer);
});

async function reload(first = false) {
  clearTimeout(timer);
  const base = `/kbs/${props.id}`;
  try {
    // ponytail: one page of up to 1000 documents; add paging if a KB ever grows past that.
    const [k, s, d] = await Promise.all([api<KB>(base), api<KbStats>(`${base}/stats`), api<Doc[]>(`${base}/documents?limit=1000`)]);
    if (!alive) return;
    kb.value = k;
    stats.value = s;
    docs.value = d;
    error.value = "";
    if (first) Object.assign(form, { name: k.name, description: k.description, status: k.status });
  } catch (e) {
    if (!alive) return;
    error.value = describeError(e);
    if (e instanceof ApiError && e.status >= 400 && e.status < 500) return; // gone or forbidden: stop polling
  } finally {
    loading.value = false;
  }
  if (alive && pending.value) timer = setTimeout(reload, 4000);
}
onMounted(() => reload(true));

async function act(fn: () => Promise<void>) {
  busy.value = true;
  actionError.value = "";
  notice.value = "";
  try {
    await fn();
  } catch (e) {
    actionError.value = describeError(e);
  } finally {
    busy.value = false;
  }
}

const save = () =>
  act(async () => {
    kb.value = await api<KB>(`/kbs/${props.id}`, { method: "PATCH", json: { name: form.name.trim(), description: form.description.trim(), status: form.status } });
    notice.value = "Perubahan disimpan.";
    loadKbs().catch(() => {});
  });

const reindexAll = () =>
  act(async () => {
    const r = await api<{ enqueued: number }>(`/kbs/${props.id}/reindex`, { method: "POST" });
    notice.value = r.enqueued ? `${r.enqueued} dokumen masuk antrean index ulang.` : "Tidak ada dokumen yang perlu diantre (semua sudah punya job aktif).";
    await reload();
  });

const removeKb = () =>
  act(async () => {
    await api(`/kbs/${props.id}`, { method: "DELETE" });
    await loadKbs().catch(() => {});
    await router.replace({ name: "kbs" });
  });
</script>

<template>
  <p><RouterLink :to="{ name: 'kbs' }">← Semua knowledge base</RouterLink></p>
  <p v-if="loading" class="muted" role="status">Memuat…</p>
  <p v-else-if="error && !kb" class="alert bad" role="alert">{{ error }}</p>

  <template v-if="kb">
    <div class="page-head">
      <div>
        <h1>{{ kb.name }}</h1>
        <p class="muted">{{ kb.description || "Tanpa deskripsi." }}</p>
        <p><StatusBadge :status="kb.status" /> <span class="badge neutral">Peranmu: {{ kb.role }}</span></p>
      </div>
      <div class="row">
        <RouterLink v-if="active" class="btn primary" :to="{ name: 'ask', query: { kb: kb.id } }">Tanya</RouterLink>
        <RouterLink v-if="editor" class="btn" :to="{ name: 'logs', query: { kb: kb.id } }">Log</RouterLink>
        <button v-if="editor && active" type="button" class="btn" :disabled="busy" @click="reindexAll">Index ulang semua</button>
      </div>
    </div>

    <p v-if="!active" class="alert info" role="status">Knowledge base ini diarsipkan. Dokumen masih bisa dibaca, tapi upload, index ulang, dan tanya dinonaktifkan.</p>
    <p v-if="error" class="alert bad" role="alert">Gagal memuat ulang: {{ error }}</p>
    <p v-if="actionError" class="alert bad" role="alert">{{ actionError }}</p>
    <p v-if="notice" class="alert ok" role="status">{{ notice }}</p>

    <section aria-labelledby="h-stats">
      <h2 id="h-stats">Ringkasan</h2>
      <dl v-if="stats" class="facts card">
        <div>
          <dt>Dokumen</dt>
          <dd>{{ stats.documents.total }} <span class="muted small">(siap {{ stats.documents.ready }}, antre {{ stats.documents.queued }}, diproses {{ stats.documents.processing }}, gagal {{ stats.documents.failed }})</span></dd>
        </div>
        <div><dt>Chunk</dt><dd>{{ stats.chunks }}</dd></div>
        <div><dt>Halaman</dt><dd>{{ stats.pages }}</dd></div>
        <div><dt>Halaman OCR</dt><dd>{{ stats.ocr_pages }}</dd></div>
        <div>
          <dt>Job index</dt>
          <dd class="small">antre {{ stats.jobs.queued }}, berjalan {{ stats.jobs.running }}, selesai {{ stats.jobs.done }}, gagal {{ stats.jobs.failed }}</dd>
        </div>
        <div><dt>Model embedding</dt><dd>{{ stats.embedding_model ?? "Belum ada" }}</dd></div>
        <div><dt>Model LLM</dt><dd>{{ stats.llm_model ?? "Belum diatur" }}</dd></div>
        <div><dt>Terakhir diindex</dt><dd>{{ fmtDate(stats.last_indexed_at, "Belum pernah") }}</dd></div>
      </dl>
      <p v-if="pending" class="muted small" role="status">Ada dokumen yang sedang diproses; daftar diperbarui otomatis.</p>
    </section>

    <section aria-labelledby="h-docs">
      <h2 id="h-docs">Dokumen</h2>
      <DocumentsPanel :kb="kb" :docs="docs" @changed="reload()" />
    </section>

    <section v-if="isAdmin" aria-labelledby="h-members">
      <h2 id="h-members">Anggota</h2>
      <MembersPanel :kb-id="kb.id" />
    </section>

    <section v-if="isAdmin" aria-labelledby="h-settings">
      <h2 id="h-settings">Pengaturan</h2>
      <form class="card" @submit.prevent="save">
        <div class="field">
          <label for="s-name">Nama</label>
          <input id="s-name" v-model="form.name" required maxlength="200" />
        </div>
        <div class="field">
          <label for="s-desc">Deskripsi</label>
          <textarea id="s-desc" v-model="form.description" rows="3" maxlength="2000"></textarea>
        </div>
        <div class="field">
          <label for="s-status">Status</label>
          <select id="s-status" v-model="form.status">
            <option value="active">Aktif</option>
            <option value="archived">Diarsipkan (hanya baca)</option>
          </select>
        </div>
        <button type="submit" class="btn primary" :disabled="busy || !form.name.trim()">Simpan</button>
      </form>
      <div class="card danger-zone">
        <h3>Hapus knowledge base</h3>
        <p class="muted">Semua dokumen, chunk, job, log kueri, dan keanggotaan ikut terhapus permanen.</p>
        <ConfirmButton label="Hapus knowledge base" warning="Hapus permanen knowledge base ini?" :disabled="busy" @confirm="removeKb" />
      </div>
    </section>
  </template>
</template>
