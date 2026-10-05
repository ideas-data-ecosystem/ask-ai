// The only place that talks to the backend. Same-origin /api, cookie session.
export class ApiError extends Error {
  status: number; // 0 = network failure
  detail: string; // server `detail` (string, or joined `msg` values for 422)
  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.detail = detail;
  }
}

// Router installs this in main.ts (api.ts must not import the router: circular).
export const hooks = { onUnauthorized: () => {} };

interface Init {
  method?: string;
  json?: unknown;
  form?: FormData;
  quiet?: boolean; // do not treat 401 as "session expired" (login, /auth/me)
}

export function parseDetail(body: unknown): string {
  const d = (body as { detail?: unknown } | null)?.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) return d.map((x) => (x as { msg?: string }).msg).filter(Boolean).join("; ");
  return "";
}

export async function api<T = void>(path: string, init: Init = {}): Promise<T> {
  const headers: Record<string, string> = {};
  let body: BodyInit | null = null;
  if (init.json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(init.json);
  } else if (init.form) {
    body = init.form; // browser sets the multipart boundary
  }
  let res: Response;
  try {
    res = await fetch("/api" + path, { method: init.method ?? "GET", headers, body, credentials: "include" });
  } catch {
    throw new ApiError(0, "");
  }
  if (res.ok) return (res.status === 204 ? undefined : await res.json()) as T;
  const json: unknown = await res.json().catch(() => null); // proxy errors may be HTML
  const err = new ApiError(res.status, parseDetail(json));
  if (res.status === 401 && !init.quiet) hooks.onUnauthorized();
  throw err;
}

// Readable Indonesian message for any thrown value.
export function describeError(e: unknown): string {
  if (!(e instanceof ApiError)) return "Terjadi kesalahan tak terduga.";
  const d = e.detail;
  switch (e.status) {
    case 0:
      return "Tidak dapat terhubung ke server. Periksa koneksi lalu coba lagi.";
    case 400:
      return d || "Permintaan tidak valid.";
    case 401:
      return "Email atau kata sandi salah.";
    case 403:
      return "Kamu tidak punya izin untuk tindakan ini.";
    case 404:
      return "Data tidak ditemukan. Mungkin sudah dihapus.";
    case 409:
      if (/archived/i.test(d)) return "Knowledge base ini diarsipkan, jadi upload, index ulang, dan tanya tidak tersedia.";
      if (/^reindex required/i.test(d)) return "Knowledge base perlu diindex ulang (model embedding berubah). Hubungi admin.";
      return d ? `Konflik: ${d}` : "Terjadi konflik dengan data yang ada.";
    case 413:
      return "File terlalu besar.";
    case 415:
      return "Tipe file tidak didukung. Gunakan pdf, docx, md, atau txt.";
    case 422:
      return d ? `Input tidak valid: ${d}` : "Input tidak valid.";
    case 429: // the contract has 429 on /ask only: the user already has 2 questions in flight
      return "Masih ada 2 pertanyaan yang diproses; tunggu salah satunya selesai.";
    case 502:
      // Only the backend's own 502s name the upstream; a dev proxy or reverse proxy 502 means the API is down.
      if (/embedding service/i.test(d)) return "Layanan embedding sedang gagal. Coba lagi sebentar lagi.";
      if (/language model/i.test(d)) return "Model bahasa sedang gagal atau menolak menjawab. Coba lagi sebentar lagi.";
      return "Server API tidak bisa dihubungi (HTTP 502). Pastikan backend berjalan.";
    default:
      return e.status >= 500 ? `Server bermasalah (HTTP ${e.status}). Coba lagi nanti.` : d || `Permintaan gagal (HTTP ${e.status}).`;
  }
}
