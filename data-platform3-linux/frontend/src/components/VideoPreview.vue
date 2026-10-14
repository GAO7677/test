<template>
  <div class="video-preview">
    <video v-if="src" :key="src" :src="src" :poster="poster || undefined"
           controls playsinline preload="metadata" @error="failed = true" />
    <el-alert v-if="failed" title="视频暂时无法播放，请确认登录状态及文件是否可用；若编码不受浏览器支持，可下载原视频查看（推荐 H.264 MP4）。"
              type="warning" :closable="false" show-icon />
    <a v-if="src" :href="src" download>下载原视频</a>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
const props = defineProps({ src: { type: String, default: '' }, poster: String })
const failed = ref(false)
watch(() => props.src, () => { failed.value = false })
</script>

<style scoped>
.video-preview { text-align: center; }
video { display: block; width: 100%; max-height: 62vh; margin: 0 auto 8px; border-radius: 8px; background: #000; }
a { display: inline-block; margin-top: 6px; font-size: 0.82rem; color: var(--el-color-primary); }
</style>
