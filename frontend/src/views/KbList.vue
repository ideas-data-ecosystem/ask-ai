<script setup lang="ts">
import { onMounted, ref } from "vue";
import { RouterLink, useRouter } from "vue-router";
import { api, describeError } from "../api.ts";
import { fmtDate } from "../lib.ts";
import { loadKbs, session } from "../session.ts";
import type { KB } from "../types.ts";
import StatusBadge from "../components/StatusBadge.vue";

const router = useRouter();
const error = ref("");
const loading = ref(true);
const creating = ref(false);
const name = ref("");
const description = ref("");
const busy = ref(false);

onMounted(async () => {
  try {
    await loadKbs();
  } catch (e) {
    error.value = describeError(e);
  } finally {
    loading.value = false;
  }
});

async function create() {
  busy.value = true;
  error.value = "";
  try {
    const kb = await api<KB>("/kbs", { method: "POST", json: { name: name.value.trim(), description: description.value.trim() } });
    await loadKbs().catch(() => {});
    await router.push({ name: "kb", params: { id: kb.id } });
  } catch (e) {
    error.value = describeError(e);
  } finally {
    busy.value = false;
  }
}
</script>

<template>
  <div class="page-head">
    <h1>Knowledge Base</h1>
    <button v-if="session.me?.is_admin && !creating" type="button" class="btn primary" @click="creating = true">Buat knowledge base</button>
  </div>

  <form v-if="creating" class="card" @submit.prevent="create">
    <h2>Knowledge base baru</h2>
    <div class="field">
      <label for="kb-name">Nama</label>
      <input id="kb-name" v-model="name" required maxlength="200" />
    </div>
    <div class="field">
      <label for="kb-desc">Deskripsi (opsional)</label>
      <textarea id="kb-desc" v-model="description" rows="2" maxlength="2000"></textarea>
    </div>
    <div class="row">
      <button type="submit" class="btn primary" :disabled="busy || !name.trim()">{{ busy ? "Menyimpan…" : "Buat" }}</button>
      <button type="button" class="btn" @click="creating = false">Batal</button>
    </div>
  </form>

  <p v-if="error" class="alert bad" role="alert">{{ error }}</p>
  <p v-if="loading" class="muted" role="status">Memuat…</p>
  <p v-else-if="session.kbs && !session.kbs.length" class="muted">
    Belum ada knowledge base yang bisa kamu akses. Minta admin untuk menambahkanmu sebagai anggota.
  </p>

  <ul class="cards">
    <li v-for="kb in session.kbs ?? []" :key="kb.id" class="card">
      <h2><RouterLink :to="{ name: 'kb', params: { id: kb.id } }">{{ kb.name }}</RouterLink></h2>
      <p class="muted">{{ kb.description || "Tanpa deskripsi." }}</p>
      <dl class="facts">
        <div><dt>Dokumen</dt><dd>{{ kb.doc_count }}</dd></div>
        <div><dt>Terakhir diindex</dt><dd>{{ fmtDate(kb.last_indexed_at, "Belum pernah") }}</dd></div>
        <div><dt>Status</dt><dd><StatusBadge :status="kb.status" /></dd></div>
        <div><dt>Peranmu</dt><dd>{{ kb.role }}</dd></div>
      </dl>
      <div class="row">
        <RouterLink v-if="kb.status === 'active'" class="btn primary sm" :to="{ name: 'ask', query: { kb: kb.id } }">Tanya</RouterLink>
        <RouterLink class="btn sm" :to="{ name: 'kb', params: { id: kb.id } }">Detail</RouterLink>
      </div>
    </li>
  </ul>
</template>
