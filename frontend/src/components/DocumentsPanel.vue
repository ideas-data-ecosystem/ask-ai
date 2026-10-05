<script setup lang="ts">
// Documents table + upload form. Loading and polling live in KbDetail; this component only
// performs actions and emits `changed` so the parent reloads.
import { computed, ref } from "vue";
import { api, ApiError, describeError } from "../api.ts";
import { fileUrl, fmtDate } from "../lib.ts";
import { canEdit } from "../session.ts";
import type { Doc, KB } from "../types.ts";
import ConfirmButton from "./ConfirmButton.vue";
import StatusBadge from "./StatusBadge.vue";

const props = defineProps<{ kb: KB; docs: Doc[] }>();
const emit = defineEmits<{ changed: [] }>();

const editable = computed(() => canEdit(props.kb));
const writable = computed(() => editable.value && props.kb.status === "active"); // archived KBs reject upload and reindex
const categories = computed(() => [...new Set(props.docs.map((d) => d.category).filter((c): c is string => !!c))]);

const files = ref<File[]>([]);
const category = ref("");
const fileInput = ref<HTMLInputElement>();
const uploading = ref(false);
const busyId = ref<string | null>(null);
const errors = ref<string[]>([]);
const notice = ref("");

const base = computed(() => `/kbs/${props.kb.id}/documents`);

function onPick(e: Event) {
  files.value = Array.from((e.target as HTMLInputElement).files ?? []);
}

async function upload() {
  uploading.value = true;
  errors.value = [];
  notice.value = "";
  let ok = 0;
  for (const f of files.value) {
    const form = new FormData();
    form.append("file", f);
    if (category.value.trim()) form.append("category", category.value.trim());
    try {
      await api(base.value, { method: "POST", form });
      ok++;
    } catch (e) {
      errors.value.push(`${f.name}: ${describeError(e)}`);
      if (e instanceof ApiError && (e.status === 401 || e.status === 403 || e.status === 0)) break; // would fail for every file
    }
  }
  if (ok) notice.value = `${ok} file diunggah dan masuk antrean index.`;
  files.value = [];
  if (fileInput.value) fileInput.value.value = "";
  uploading.value = false;
  emit("changed");
}

async function act(d: Doc, run: () => Promise<unknown>, done: string) {
  busyId.value = d.id;
  errors.value = [];
  notice.value = "";
  try {
    await run();
    notice.value = done;
  } catch (e) {
    errors.value = [`${d.filename}: ${describeError(e)}`];
  } finally {
    busyId.value = null;
    emit("changed");
  }
}
const reindex = (d: Doc) => act(d, () => api(`${base.value}/${d.id}/reindex`, { method: "POST" }), `${d.filename} masuk antrean index ulang.`);
const remove = (d: Doc) => act(d, () => api(`${base.value}/${d.id}`, { method: "DELETE" }), `${d.filename} dihapus.`);
</script>

<template>
  <form v-if="writable" class="card" @submit.prevent="upload">
    <h3>Unggah dokumen</h3>
    <div class="field">
      <label for="up-file">File (pdf, docx, md, txt; boleh lebih dari satu)</label>
      <input id="up-file" ref="fileInput" type="file" multiple accept=".pdf,.docx,.md,.txt" required @change="onPick" />
    </div>
    <div class="field">
      <label for="up-cat">Kategori (opsional)</label>
      <input id="up-cat" v-model="category" maxlength="200" list="up-cats" />
      <datalist id="up-cats"><option v-for="c in categories" :key="c" :value="c" /></datalist>
    </div>
    <button type="submit" class="btn primary" :disabled="uploading || !files.length">{{ uploading ? "Mengunggah…" : "Unggah" }}</button>
  </form>

  <p v-if="notice" class="alert ok" role="status">{{ notice }}</p>
  <div v-if="errors.length" class="alert bad" role="alert">
    <p v-for="m in errors" :key="m">{{ m }}</p>
  </div>

  <p v-if="!docs.length" class="muted">Belum ada dokumen di knowledge base ini.</p>
  <table v-else class="table">
    <thead>
      <tr>
        <th scope="col">Berkas</th>
        <th scope="col">Kategori</th>
        <th scope="col">Status</th>
        <th scope="col">Halaman</th>
        <th scope="col">Hal. OCR</th>
        <th scope="col">Chunk</th>
        <th scope="col">Aksi</th>
      </tr>
    </thead>
    <tbody>
      <tr v-for="d in docs" :key="d.id">
        <td data-label="Berkas">
          <div>
            <a :href="fileUrl(kb.id, d.id)" target="_blank" rel="noopener">{{ d.filename }}</a>
            <div v-if="d.status === 'failed' && d.error" class="error-text">Galat: {{ d.error }}</div>
            <div class="muted small">Diindex: {{ fmtDate(d.indexed_at, "belum") }}</div>
          </div>
        </td>
        <td data-label="Kategori">{{ d.category ?? "-" }}</td>
        <td data-label="Status"><StatusBadge :status="d.status" /></td>
        <td data-label="Halaman">{{ d.page_count ?? "-" }}</td>
        <td data-label="Hal. OCR">{{ d.ocr_pages }}</td>
        <td data-label="Chunk">{{ d.chunk_count }}</td>
        <td data-label="Aksi">
          <span v-if="editable" class="row">
            <button
              v-if="writable"
              type="button"
              class="btn sm"
              :disabled="busyId === d.id || d.status === 'queued' || d.status === 'processing'"
              :aria-label="`${d.status === 'failed' ? 'Coba lagi' : 'Index ulang'} ${d.filename}`"
              @click="reindex(d)"
            >
              {{ d.status === "failed" ? "Coba lagi" : "Index ulang" }}
            </button>
            <ConfirmButton label="Hapus" :aria-label="`Hapus ${d.filename}`" :disabled="busyId === d.id" warning="Hapus dokumen ini?" @confirm="remove(d)" />
          </span>
          <span v-else class="muted">-</span>
        </td>
      </tr>
    </tbody>
  </table>
</template>
