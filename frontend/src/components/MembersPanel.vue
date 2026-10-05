<script setup lang="ts">
// Admin only: the contract restricts every member endpoint to admins.
import { computed, onMounted, ref } from "vue";
import { api, describeError } from "../api.ts";
import type { Member, User } from "../types.ts";
import ConfirmButton from "./ConfirmButton.vue";

const props = defineProps<{ kbId: string }>();

const members = ref<Member[]>([]);
const users = ref<User[]>([]);
const error = ref("");
const loading = ref(true);
const busy = ref(false);
const newUser = ref("");
const newRole = ref<Member["role"]>("viewer");

const candidates = computed(() => users.value.filter((u) => u.is_active && !members.value.some((m) => m.user_id === u.id)));
const url = (userId: string) => `/kbs/${props.kbId}/members/${userId}`;

async function load() {
  try {
    [members.value, users.value] = await Promise.all([api<Member[]>(`/kbs/${props.kbId}/members`), api<User[]>("/users")]);
  } catch (e) {
    error.value = describeError(e);
  } finally {
    loading.value = false;
  }
}
onMounted(load);

async function run(fn: () => Promise<unknown>) {
  busy.value = true;
  error.value = "";
  try {
    await fn();
  } catch (e) {
    error.value = describeError(e);
  } finally {
    busy.value = false;
    await load();
  }
}
const setRole = (userId: string, role: Member["role"]) => run(() => api(url(userId), { method: "PUT", json: { role } }));
const remove = (m: Member) => run(() => api(url(m.user_id), { method: "DELETE" }));
async function add() {
  await run(() => api(url(newUser.value), { method: "PUT", json: { role: newRole.value } }));
  newUser.value = "";
}
</script>

<template>
  <p v-if="error" class="alert bad" role="alert">{{ error }}</p>
  <p v-if="loading" class="muted" role="status">Memuat anggota…</p>
  <template v-else>
    <p class="muted small">Admin otomatis punya akses ke semua knowledge base dan tidak perlu didaftarkan.</p>
    <p v-if="!members.length" class="muted">Belum ada anggota.</p>
    <table v-else class="table">
      <thead>
        <tr><th scope="col">Nama</th><th scope="col">Email</th><th scope="col">Peran</th><th scope="col">Aksi</th></tr>
      </thead>
      <tbody>
        <tr v-for="m in members" :key="m.user_id">
          <td data-label="Nama">{{ m.name }}</td>
          <td data-label="Email">{{ m.email }}</td>
          <td data-label="Peran">
            <select :aria-label="`Peran ${m.name}`" :value="m.role" :disabled="busy" @change="setRole(m.user_id, ($event.target as HTMLSelectElement).value as Member['role'])">
              <option value="viewer">viewer</option>
              <option value="editor">editor</option>
            </select>
          </td>
          <td data-label="Aksi">
            <ConfirmButton label="Keluarkan" :aria-label="`Keluarkan ${m.name}`" warning="Keluarkan anggota ini?" :disabled="busy" @confirm="remove(m)" />
          </td>
        </tr>
      </tbody>
    </table>

    <form class="row form-row" @submit.prevent="add">
      <div class="field">
        <label for="m-user">Tambah anggota</label>
        <select id="m-user" v-model="newUser" required>
          <option value="" disabled>Pilih pengguna…</option>
          <option v-for="u in candidates" :key="u.id" :value="u.id">{{ u.name }} ({{ u.email }})</option>
        </select>
      </div>
      <div class="field">
        <label for="m-role">Peran</label>
        <select id="m-role" v-model="newRole">
          <option value="viewer">viewer</option>
          <option value="editor">editor</option>
        </select>
      </div>
      <button type="submit" class="btn primary" :disabled="busy || !newUser">Tambah</button>
    </form>
  </template>
</template>
