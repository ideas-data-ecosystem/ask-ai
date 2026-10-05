<script setup lang="ts">
// Small menu (profile, history item, knowledge-base picker). The trigger comes from the parent through the
// `trigger` slot; the menu is teleported to <body> so scrolling sidebars and transformed drawers cannot clip it.
// Keyboard: Esc closes and returns focus to the trigger, Arrow/Home/End move between items, Tab closes.
import { computed, nextTick, onBeforeUnmount, ref, useId } from "vue";

const props = withDefaults(defineProps<{ label: string; placement?: "top" | "bottom"; align?: "start" | "end"; matchWidth?: boolean }>(), {
  placement: "bottom",
  align: "start",
  matchWidth: false,
});

const open = ref(false);
const root = ref<HTMLElement>();
const menu = ref<HTMLElement>();
const pos = ref<Record<string, string>>({});
const menuId = useId();
const triggerAttrs = computed(() => ({ "aria-haspopup": "menu" as const, "aria-expanded": open.value, "aria-controls": open.value ? menuId : undefined }));

const items = () => Array.from(menu.value?.querySelectorAll<HTMLElement>('[role^="menuitem"]:not(:disabled)') ?? []);
const trigger = () => root.value?.querySelector<HTMLElement>("button, a[href]");

function place() {
  const r = root.value!.getBoundingClientRect();
  const s: Record<string, string> = {};
  if (props.placement === "top") s.bottom = `${window.innerHeight - r.top + 6}px`;
  else s.top = `${r.bottom + 6}px`;
  if (props.align === "end") s.right = `${window.innerWidth - r.right}px`;
  else s.left = `${r.left}px`;
  if (props.matchWidth) s.minWidth = `${r.width}px`;
  pos.value = s;
}

function onDocPointer(e: Event) {
  const t = e.target as Node;
  if (!root.value?.contains(t) && !menu.value?.contains(t)) close(false);
}
// Capture phase: Esc closes the menu first and never reaches the drawer's own Esc handler.
function onDocKey(e: KeyboardEvent) {
  if (e.key !== "Escape") return;
  e.preventDefault();
  e.stopPropagation();
  close();
}
function onViewport(e: Event) {
  if (!menu.value?.contains(e.target as Node)) close(false); // a long menu may scroll itself
}

async function show() {
  place();
  open.value = true;
  document.addEventListener("pointerdown", onDocPointer);
  document.addEventListener("keydown", onDocKey, true);
  window.addEventListener("resize", onViewport);
  window.addEventListener("scroll", onViewport, true);
  await nextTick();
  const list = items();
  (list.find((i) => i.getAttribute("aria-checked") === "true") ?? list[0])?.focus();
}

function close(refocus = true) {
  if (!open.value) return;
  open.value = false;
  document.removeEventListener("pointerdown", onDocPointer);
  document.removeEventListener("keydown", onDocKey, true);
  window.removeEventListener("resize", onViewport);
  window.removeEventListener("scroll", onViewport, true);
  if (refocus) trigger()?.focus();
}

const toggle = () => (open.value ? close() : show());
onBeforeUnmount(() => close(false));

function onMenuKey(e: KeyboardEvent) {
  const list = items();
  const i = list.indexOf(document.activeElement as HTMLElement);
  const go = (n: number) => {
    e.preventDefault();
    list[(n + list.length) % list.length]?.focus();
  };
  if (e.key === "ArrowDown") go(i + 1);
  else if (e.key === "ArrowUp") go(i - 1);
  else if (e.key === "Home") go(0);
  else if (e.key === "End") go(list.length - 1);
  else if (e.key === "Tab") {
    e.preventDefault();
    close();
  }
}
</script>

<template>
  <div ref="root" class="dd">
    <slot name="trigger" :toggle="toggle" :attrs="triggerAttrs"></slot>
    <Teleport to="body">
      <div v-if="open" :id="menuId" ref="menu" class="menu" role="menu" :aria-label="label" :style="pos" @keydown="onMenuKey">
        <slot :close="close"></slot>
      </div>
    </Teleport>
  </div>
</template>
