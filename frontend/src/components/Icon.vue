<script setup lang="ts">
// Monochrome SVG icon from src/assets, drawn as a CSS mask so it follows `color`.
// Decorative: every icon-only button carries its own aria-label.
import { computed } from "vue";

const props = defineProps<{ src: string; size?: number }>();
// Vite inlines small SVGs as data URIs that contain single quotes, so the url() needs double quotes.
const style = computed(() => ({ "--icon": `url("${props.src}")`, ...(props.size ? { "--icon-size": `${props.size}px` } : {}) }));
</script>

<template>
  <span class="icon" aria-hidden="true" :style="style"></span>
</template>
