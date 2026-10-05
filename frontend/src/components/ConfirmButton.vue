<script setup lang="ts">
// Inline two-step confirmation (replaces window.confirm). Focus moves to "Batal" when asking
// and back to the trigger when cancelled.
import { nextTick, ref } from "vue";

defineProps<{ label: string; warning?: string; disabled?: boolean; ariaLabel?: string }>();
defineEmits<{ confirm: [] }>();

const asking = ref(false);
const trigger = ref<HTMLButtonElement>();
const cancel = ref<HTMLButtonElement>();

async function ask() {
  asking.value = true;
  await nextTick();
  cancel.value?.focus();
}
async function stop() {
  asking.value = false;
  await nextTick();
  trigger.value?.focus();
}
</script>

<template>
  <button v-if="!asking" ref="trigger" type="button" class="btn sm danger-outline" :disabled="disabled" :aria-label="ariaLabel" @click="ask">
    {{ label }}
  </button>
  <span v-else class="confirm" role="group" :aria-label="ariaLabel ?? label" @keydown.esc="stop">
    <span>{{ warning ?? "Yakin?" }}</span>
    <button type="button" class="btn sm danger" @click="asking = false; $emit('confirm')">Ya, hapus</button>
    <button ref="cancel" type="button" class="btn sm" @click="stop">Batal</button>
  </span>
</template>
