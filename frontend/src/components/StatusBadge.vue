<script setup lang="ts">
// Status is shown as icon + text, so it never depends on color alone.
import { computed } from "vue";

const props = defineProps<{ status: string }>();

const MAP: Record<string, { label: string; icon: string; tone: string }> = {
  queued: { label: "Antre", icon: "…", tone: "neutral" },
  processing: { label: "Diproses", icon: "↻", tone: "info" },
  running: { label: "Berjalan", icon: "↻", tone: "info" },
  ready: { label: "Siap", icon: "✓", tone: "ok" },
  done: { label: "Selesai", icon: "✓", tone: "ok" },
  failed: { label: "Gagal", icon: "✕", tone: "bad" },
  active: { label: "Aktif", icon: "✓", tone: "ok" },
  inactive: { label: "Nonaktif", icon: "⊘", tone: "neutral" },
  archived: { label: "Diarsipkan", icon: "▤", tone: "neutral" },
};
const s = computed(() => MAP[props.status] ?? { label: props.status, icon: "•", tone: "neutral" });
</script>

<template>
  <span class="badge" :class="s.tone"><span aria-hidden="true">{{ s.icon }}</span> {{ s.label }}</span>
</template>
