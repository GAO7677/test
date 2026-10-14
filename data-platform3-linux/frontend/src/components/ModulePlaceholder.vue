<template>
  <div class="page">
    <div class="page-header">
      <h2><span>{{ icon }}</span> {{ title }}</h2>
      <p>{{ description }}</p>
    </div>

    <div class="table-card placeholder-card">
      <el-empty :description="emptyDescription" :image-size="88">
        <el-tag type="info" effect="plain" round>功能建设中</el-tag>
      </el-empty>
    </div>
  </div>
</template>

<script setup>
defineProps({
  title: { type: String, required: true },
  icon: { type: String, default: '🧩' },
  description: { type: String, default: '' },
  emptyDescription: { type: String, default: '功能建设中' },
})
</script>

<style scoped>
.placeholder-card {
  min-height: 360px;
  display: grid;
  place-items: center;
  background:
    radial-gradient(circle at 50% 0%, rgba(96, 165, 250, 0.08), transparent 42%),
    var(--dp-bg-card);
}
.placeholder-card :deep(.el-empty) { padding: 36px 20px; }
.placeholder-card :deep(.el-empty__description p) { color: var(--dp-text-secondary); }
</style>