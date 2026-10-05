<script setup lang="ts">
// The prompt pill: knowledge-base picker, a textarea that grows with its content, and the send button.
// Enter sends, Shift+Enter adds a newline. While a question is pending the field is read-only (it keeps focus).
import { computed, nextTick, onMounted, ref, watch } from "vue";
import type { KB } from "../types.ts";
import DropdownMenu from "./DropdownMenu.vue";
import Icon from "./Icon.vue";
import checkIcon from "../assets/check.svg";
import chevronIcon from "../assets/chevron-down.svg";
import sendIcon from "../assets/send.svg";

const props = defineProps<{ modelValue: string; kbs: KB[]; kbId: string; busy: boolean; maxLength: number }>();
const emit = defineEmits<{ "update:modelValue": [string]; "update:kbId": [string]; submit: [] }>();

const field = ref<HTMLTextAreaElement>();
const current = computed(() => props.kbs.find((k) => k.id === props.kbId));
const canSend = computed(() => current.value?.status === "active" && !!props.modelValue.trim() && !props.busy);

function fit() {
  const el = field.value;
  if (!el) return;
  el.style.height = "auto";
  el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
}
onMounted(fit);
watch(() => props.modelValue, () => nextTick(fit)); // typing, prefill and clearing after send

function onKey(e: KeyboardEvent) {
  if (e.key !== "Enter" || e.shiftKey || e.isComposing) return;
  e.preventDefault();
  if (canSend.value) emit("submit");
}

defineExpose({ focus: () => field.value?.focus() });
</script>

<template>
  <div class="composer-box">
  <form class="composer" :class="{ 'is-busy': busy }" @submit.prevent="canSend && emit('submit')">
    <DropdownMenu label="Pilih knowledge base" placement="top" align="start">
      <template #trigger="{ attrs, toggle }">
        <button type="button" class="kb-pill" v-bind="attrs" :disabled="busy" @click="toggle">
          <span class="sr-only">Knowledge base:</span>
          <span class="kb-pill-name">{{ current?.name ?? "Pilih" }}</span>
          <Icon :src="chevronIcon" :size="10" />
        </button>
      </template>
      <template #default="{ close }">
        <button v-for="k in kbs" :key="k.id" type="button" role="menuitemradio" class="menu-item" :aria-checked="k.id === kbId" :disabled="k.status !== 'active'" @click="close(); emit('update:kbId', k.id)">
          <span class="menu-item-text">{{ k.name }}<small v-if="k.status !== 'active'"> (diarsipkan)</small></span>
          <Icon v-if="k.id === kbId" :src="checkIcon" :size="14" />
        </button>
      </template>
    </DropdownMenu>

    <textarea
      ref="field"
      :value="modelValue"
      rows="1"
      :maxlength="maxLength"
      :readonly="busy"
      aria-label="Pertanyaan"
      placeholder="Tanyakan sesuatu tentang dokumen di knowledge base ini..."
      @input="emit('update:modelValue', ($event.target as HTMLTextAreaElement).value)"
      @keydown="onKey"
    ></textarea>

    <button type="submit" class="send" :disabled="!canSend" aria-label="Kirim pertanyaan">
      <Icon :src="sendIcon" :size="22" />
    </button>
  </form>
  <p class="composer-hint">
    <span class="hint-keys">Enter untuk kirim, Shift+Enter untuk baris baru</span>
    <span v-if="modelValue.length >= maxLength * 0.8" class="hint-count" :class="{ full: modelValue.length >= maxLength }">{{ modelValue.length }} / {{ maxLength }}</span>
  </p>
  </div>
</template>
