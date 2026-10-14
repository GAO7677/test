<template>
  <!-- 任务列表：进度轮询刷新（2秒，仅当存在进行中任务） -->
  <div>
    <el-table :data="rows" v-loading="loading" size="default">
      <slot :rows="rows" />
      <el-table-column label="状态" width="110">
        <template #default="{ row }">
          <el-tag size="small" effect="dark" :type="statusTag(row.status)">{{ statusName(row.status) }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column v-if="hasProgress" label="进度" width="150">
        <template #default="{ row }">
          <el-progress v-if="row.progress != null" :percentage="row.progress"
                       :status="row.status === 'failed' ? 'exception' : row.status === 'completed' ? 'success' : ''" />
          <span v-else class="muted">—</span>
        </template>
      </el-table-column>
      <el-table-column label="版本" width="150">
        <template #default="{ row }">
          <span v-if="versionMap[row.input_version_id]" class="muted">输入 {{ versionMap[row.input_version_id] }}</span>
          <span v-if="versionMap[row.output_version_id]" style="color: #34d399"> → {{ versionMap[row.output_version_id] }}</span>
          <span v-if="!versionMap[row.input_version_id] && !versionMap[row.output_version_id]" class="muted">—</span>
        </template>
      </el-table-column>
      <el-table-column prop="created_at" label="创建时间" width="160" />
      <el-table-column label="错误" min-width="120">
        <template #default="{ row }">
          <span v-if="row.error_msg" style="color: #ef4444; font-size: 0.78rem">{{ row.error_msg }}</span>
        </template>
      </el-table-column>
      <template #empty><div class="empty-tip">暂无任务</div></template>
    </el-table>
  </div>
</template>

<script setup>
import { onBeforeUnmount, onMounted, ref } from 'vue'
import api from '../api'

const props = defineProps({
  url: { type: String, required: true },
  hasProgress: { type: Boolean, default: true },
  versionMap: { type: Object, default: () => ({}) },  // version_id -> "V{n}"，由父页面提供
})
const rows = ref([])
const loading = ref(false)
let timer = null

function statusTag(s) {
  return { pending: 'info', running: 'primary', completed: 'success', failed: 'danger' }[s] || 'info'
}
function statusName(s) {
  return { pending: '排队中', running: '进行中', completed: '已完成', failed: '失败', cancelled: '已取消' }[s] || s
}

async function load(silent) {
  if (!silent) loading.value = true
  try {
    const data = await api.get(props.url, { params: { page: 1, page_size: 50 }, silent: true })
    rows.value = data.items
    const active = data.items.some((r) => ['pending', 'running'].includes(r.status))
    clearTimeout(timer)
    if (active) timer = setTimeout(() => load(true), 2000)  // 轮询刷新
  } finally { loading.value = false }
}

onMounted(() => load())
onBeforeUnmount(() => clearTimeout(timer))
defineExpose({ reload: () => load(true) })
</script>

<style scoped>
.muted { color: #64748b; }
.empty-tip { text-align: center; color: #64748b; padding: 30px 0; }
</style>
