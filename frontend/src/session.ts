// Shared app state: current user and the KB list the user can see. No state library needed.
import { reactive } from "vue";
import { api, ApiError } from "./api.ts";
import { wipeHistory } from "./lib.ts";
import type { KB, User } from "./types.ts";

export const session = reactive<{ me: User | null; loaded: boolean; kbs: KB[] | null }>({
  me: null,
  loaded: false,
  kbs: null,
});

export async function loadMe(): Promise<void> {
  try {
    session.me = await api<User>("/auth/me", { quiet: true });
  } catch (e) {
    // Any failure on boot lands on the login page; the login form shows the real error.
    session.me = null;
    if (!(e instanceof ApiError)) throw e;
    // An expired session on reload never changes the user id, so the chat watcher would not wipe history.
    if (e.status === 401) wipeHistory(() => sessionStorage);
  }
  session.loaded = true;
}

export async function login(email: string, password: string): Promise<void> {
  session.kbs = null; // never show the previous user's KBs
  session.me = await api<User>("/auth/login", { method: "POST", json: { email, password }, quiet: true });
}

export async function logout(): Promise<void> {
  try {
    await api("/auth/logout", { method: "POST", quiet: true });
  } finally {
    session.me = null;
    session.kbs = null;
  }
}

export async function loadKbs(): Promise<KB[]> {
  const kbs = await api<KB[]>("/kbs");
  session.kbs = kbs;
  return kbs;
}

export const canEdit = (kb: KB) => kb.role === "admin" || kb.role === "editor";

// Shown on the profile card: admins are "Admin", otherwise the highest KB role the user holds.
export function roleLabel(): string {
  if (session.me?.is_admin) return "Admin";
  if (!session.kbs) return "Pengguna";
  return session.kbs.some((k) => k.role === "editor") ? "Editor" : "Viewer";
}
