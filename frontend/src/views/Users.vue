<script setup lang="ts">
import { nextTick, onMounted, reactive, ref } from "vue";
import { useRouter } from "vue-router";
import { api, describeError } from "../api.ts";
import { fmtDate } from "../lib.ts";
import { logout, session } from "../session.ts";
import type { User } from "../types.ts";
import StatusBadge from "../components/StatusBadge.vue";

const router = useRouter();
const users = ref<User[]>([]);
const loading = ref(true);
const error = ref("");
const notice = ref("");
const busy = ref(false);

const creating = ref(false);
const nu = reactive({ email: "", name: "", password: "", is_admin: false });
const editing = ref<User | null>(null);
const editName = ref<HTMLInputElement>();
const ed = reactive({ name: "", is_active: true, is_admin: false, password: "" });

const isSelf = (u: User) => u.id === session.me?.id;
const isLocked = (u: User) => !!u.locked_until && new Date(u.locked_until) > new Date();

async function load() {
  try {
    users.value = await api<User[]>("/users");
  } catch (e) {
    error.value = describeError(e);
  } finally {
    loading.value = false;
  }
}
onMounted(load);

async function run(fn: () => Promise<void>) {
  busy.value = true;
  error.value = "";
  notice.value = "";
  try {
    await fn();
  } catch (e) {
    error.value = describeError(e);
  } finally {
    busy.value = false;
  }
}

const create = () =>
  run(async () => {
    await api<User>("/users", { method: "POST", json: { email: nu.email.trim(), name: nu.name.trim(), password: nu.password, is_admin: nu.is_admin } });
    notice.value = `Pengguna ${nu.email.trim()} dibuat.`;
    Object.assign(nu, { email: "", name: "", password: "", is_admin: false });
    creating.value = false;
    await load();
  });

async function edit(u: User) {
  editing.value = u;
  Object.assign(ed, { name: u.name, is_active: u.is_active, is_admin: u.is_admin, password: "" });
  notice.value = "";
  await nextTick();
  editName.value?.focus(); // the form sits above the table: move focus (and scroll) to it
}

const save = () =>
  run(async () => {
    const u = editing.value;
    if (!u) return;
    // Send only what changed; omitted fields stay as they are.
    const json: Record<string, unknown> = {};
    if (ed.name.trim() !== u.name) json.name = ed.name.trim();
    if (ed.is_active !== u.is_active) json.is_active = ed.is_active;
    if (ed.is_admin !== u.is_admin) json.is_admin = ed.is_admin;
    if (ed.password) json.password = ed.password;
    if (!Object.keys(json).length) {
      editing.value = null;
      return;
    }
    const updated = await api<User>(`/users/${u.id}`, { method: "PATCH", json });
    editing.value = null;
    if (isSelf(u) && ed.password) {
      // Changing your own password deletes your sessions: sign out cleanly.
      await logout().catch(() => {});
      await router.replace({ name: "login" });
      return;
    }
    if (isSelf(u)) session.me = updated;
    notice.value = `Perubahan untuk ${updated.email} disimpan.${ed.password ? " Sesi pengguna itu dihentikan." : ""}`;
    await load();
  });
</script>

<template>
  <div class="page-head">
    <h1>Pengguna</h1>
    <button v-if="!creating" type="button" class="btn primary" @click="creating = true">Tambah pengguna</button>
  </div>

  <form v-if="creating" class="card" @submit.prevent="create">
    <h2>Pengguna baru</h2>
    <div class="field">
      <label for="n-email">Email</label>
      <input id="n-email" v-model="nu.email" type="email" required autocomplete="off" />
    </div>
    <div class="field">
      <label for="n-name">Nama</label>
      <input id="n-name" v-model="nu.name" required maxlength="200" />
    </div>
    <div class="field">
      <label for="n-pass">Kata sandi (8-128 karakter)</label>
      <input id="n-pass" v-model="nu.password" type="password" required minlength="8" maxlength="128" autocomplete="new-password" />
    </div>
    <label class="check"><input v-model="nu.is_admin" type="checkbox" /> Jadikan admin</label>
    <div class="row">
      <button type="submit" class="btn primary" :disabled="busy">Buat</button>
      <button type="button" class="btn" @click="creating = false">Batal</button>
    </div>
  </form>

  <form v-if="editing" class="card" @submit.prevent="save">
    <h2>Ubah {{ editing.email }}</h2>
    <div class="field">
      <label for="e-name">Nama</label>
      <input id="e-name" ref="editName" v-model="ed.name" required maxlength="200" />
    </div>
    <label class="check"><input v-model="ed.is_active" type="checkbox" :disabled="isSelf(editing)" /> Aktif</label>
    <label class="check"><input v-model="ed.is_admin" type="checkbox" :disabled="isSelf(editing)" /> Admin</label>
    <p v-if="isSelf(editing)" class="muted small">Status aktif dan admin akunmu sendiri tidak bisa diubah di sini.</p>
    <div class="field">
      <label for="e-pass">Kata sandi baru (kosongkan jika tidak diubah)</label>
      <input id="e-pass" v-model="ed.password" type="password" minlength="8" maxlength="128" autocomplete="new-password" />
      <span class="muted small">Mengganti kata sandi menghentikan semua sesi pengguna itu dan membuka kuncinya.</span>
    </div>
    <div class="row">
      <button type="submit" class="btn primary" :disabled="busy || !ed.name.trim()">Simpan</button>
      <button type="button" class="btn" @click="editing = null">Batal</button>
    </div>
  </form>

  <p v-if="notice" class="alert ok" role="status">{{ notice }}</p>
  <p v-if="error" class="alert bad" role="alert">{{ error }}</p>
  <p v-if="loading" class="muted" role="status">Memuat…</p>

  <table v-if="users.length" class="table">
    <thead>
      <tr><th scope="col">Nama</th><th scope="col">Email</th><th scope="col">Status</th><th scope="col">Peran</th><th scope="col">Dibuat</th><th scope="col">Aksi</th></tr>
    </thead>
    <tbody>
      <tr v-for="u in users" :key="u.id">
        <td data-label="Nama">{{ u.name }}<span v-if="isSelf(u)" class="muted small"> (kamu)</span></td>
        <td data-label="Email">{{ u.email }}</td>
        <td data-label="Status">
          <StatusBadge :status="u.is_active ? 'active' : 'inactive'" />
          <span v-if="isLocked(u)" class="badge warn"><span aria-hidden="true">⚠</span> Terkunci sampai {{ fmtDate(u.locked_until) }}</span>
        </td>
        <td data-label="Peran">{{ u.is_admin ? "Admin" : "Pengguna" }}</td>
        <td data-label="Dibuat">{{ fmtDate(u.created_at) }}</td>
        <td data-label="Aksi"><button type="button" class="btn sm" :aria-label="`Ubah ${u.email}`" @click="edit(u)">Ubah</button></td>
      </tr>
    </tbody>
  </table>
</template>
