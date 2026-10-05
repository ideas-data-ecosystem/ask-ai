<script setup lang="ts">
import { ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import { describeError } from "../api.ts";
import { login } from "../session.ts";
import Icon from "../components/Icon.vue";
import badge from "../assets/badge.svg";
import eyeIcon from "../assets/eye.svg";
import eyeOffIcon from "../assets/eye-off.svg";
import hero from "../assets/hero.png";
import logo from "../assets/logo.png";

const route = useRoute();
const router = useRouter();
const email = ref("");
const password = ref("");
const error = ref("");
const busy = ref(false);
const show = ref(false); // password visibility

async function submit() {
  busy.value = true;
  error.value = "";
  try {
    await login(email.value, password.value);
    const to = route.query.redirect;
    // Only follow same-site paths (guards against open redirects).
    await router.replace(typeof to === "string" && to.startsWith("/") && !to.startsWith("//") ? to : { name: "kbs" });
  } catch (e) {
    error.value = describeError(e);
  } finally {
    busy.value = false;
  }
}
</script>

<template>
  <div class="auth">
    <figure class="auth-hero">
      <img class="auth-hero-img" :src="hero" alt="" />
      <span class="auth-badge"><img :src="badge" alt="" width="36" height="36" /></span>
    </figure>
    <main id="main" class="auth-panel" tabindex="-1">
      <form class="auth-form" @submit.prevent="submit">
        <img class="auth-logo" :src="logo" alt="" width="78" height="90" />
        <h1>Masuk ke IDEAS Ask</h1>
        <p class="auth-sub">Masuk untuk bertanya ke knowledge base peraturan.</p>

        <div class="auth-fields">
          <label for="email" class="sr-only">Email</label>
          <input id="email" v-model="email" type="email" placeholder="Masukkan alamat email" autocomplete="username" required autofocus />
          <label for="password" class="sr-only">Kata sandi</label>
          <div class="pw">
            <input id="password" v-model="password" :type="show ? 'text' : 'password'" placeholder="Kata sandi" autocomplete="current-password" required />
            <button type="button" class="icon-btn pw-toggle" :aria-label="show ? 'Sembunyikan kata sandi' : 'Tampilkan kata sandi'" @click="show = !show">
              <Icon :src="show ? eyeIcon : eyeOffIcon" :size="20" />
            </button>
          </div>
          <p v-if="error" class="alert bad" role="alert">{{ error }}</p>
          <button type="submit" class="btn primary lg" :disabled="busy">{{ busy ? "Masuk…" : "Masuk" }}</button>
        </div>
      </form>
    </main>
  </div>
</template>
