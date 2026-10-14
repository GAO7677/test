<template>
  <!-- 输入版本下拉（8.7）：选项为数据集全部版本，draft 标"草稿"、published 标"已发布"，默认当前版本 -->
  <el-select :model-value="modelValue" @update:model-value="$emit('update:modelValue', $event)"
             :loading="loading" filterable style="width: 100%" placeholder="选择输入版本">
    <el-option v-for="v in versions" :key="v.id" :value="v.id" :disabled="v.ready === false"
               :label="`${v.version}（${v.stage} · ${v.ready === false ? '未完成或失败' : (v.status === 'published' ? '已发布' : '草稿')}）`">
      <span>{{ v.version }}</span>
      <el-tag size="small" effect="plain" :type="v.status === 'published' ? 'success' : 'warning'" style="margin-left: 8px">
        {{ v.ready === false ? '未完成或失败' : (v.status === 'published' ? '已发布' : '草稿') }}
      </el-tag>
      <span style="float: right; color: #64748b; font-size: 0.75rem">{{ v.stage }}</span>
    </el-option>
  </el-select>
</template>

<script setup>
import { ref, watch } from 'vue'
import api from '../api'

const props = defineProps({
  datasetId: [Number, String],
  modelValue: [Number, String],
  currentVersionId: [Number, String],
})
defineEmits(['update:modelValue'])

const versions = ref([])
const loading = ref(false)

async function load() {
  if (!props.datasetId) { versions.value = []; return }
  loading.value = true
  try {
    const data = await api.get(`/datasets/${props.datasetId}/versions`)
    versions.value = [...data.items].sort((a, b) => a.version_no - b.version_no)
  } finally { loading.value = false }
}
watch(() => props.datasetId, load, { immediate: true })
</script>
