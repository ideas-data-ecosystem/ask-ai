<script setup lang="ts">
// Empty-state cards built from the selected knowledge base's real documents (see suggestionGroups in lib.ts).
import type { SuggestionGroup } from "../lib.ts";
import Icon from "./Icon.vue";
import docIcon from "../assets/doc.svg";
import directionIcon from "../assets/icon-direction.svg";
import educationIcon from "../assets/icon-education.svg";
import noteIcon from "../assets/icon-note.svg";
import presentationIcon from "../assets/icon-presentation.svg";

defineProps<{ groups: SuggestionGroup[] }>();
defineEmits<{ pick: [title: string] }>();

const THEMES = [
  { icon: educationIcon, tone: "purple" },
  { icon: noteIcon, tone: "blue" },
  { icon: presentationIcon, tone: "green" },
  { icon: directionIcon, tone: "orange" },
];
</script>

<template>
  <ul class="sug-grid">
    <li v-for="(g, i) in groups" :key="g.category" class="sug" :class="`tone-${THEMES[i % THEMES.length]!.tone}`">
      <h2 class="sug-head">
        <span class="sug-icon"><img :src="THEMES[i % THEMES.length]!.icon" alt="" width="20" height="20" /></span>
        <span class="sug-title">{{ g.category }}</span>
      </h2>
      <ul class="sug-list">
        <li v-for="t in g.titles" :key="t">
          <button type="button" class="sug-item" :title="t" @click="$emit('pick', t)">
            <Icon :src="docIcon" :size="24" />
            <span>{{ t }}</span>
          </button>
        </li>
      </ul>
    </li>
  </ul>
</template>
