<template>
  <!-- 版本历史弹窗（8.7）：表格 + 发布/删除/下载/血缘；血缘视图为链式文本展示，不引入图表库 -->
  <el-dialog v-model="visible" :title="`版本历史 — ${dataset?.name || ''}`" width="900px" @open="load">
    <el-tabs v-model="tab" @tab-change="onTabChange">
      <el-tab-pane label="版本列表" name="list">
        <el-table :data="versions" v-loading="loading" size="default">
          <el-table-column label="版本号" width="110">
            <template #default="{ row }">
              <el-tag effect="dark" :type="row.status === 'published' ? 'success' : 'warning'">
                {{ row.version }} · {{ row.status === 'published' ? '已发布' : '草稿' }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="stage" label="阶段" width="90" />
          <el-table-column label="来源任务" width="140">
            <template #default="{ row }">
              {{ taskName(row.source_task_type) }}<span v-if="row.source_task_id" class="muted"> #{{ row.source_task_id }}</span>
            </template>
          </el-table-column>
          <el-table-column label="质量分" width="90">
            <template #default="{ row }">
              <span v-if="row.quality_score != null" :style="{ color: row.quality_score >= 80 ? '#34d399' : row.quality_score >= 60 ? '#fbbf24' : '#ef4444' }">
                {{ row.quality_score }}
              </span>
              <span v-else class="muted">未评估</span>
            </template>
          </el-table-column>
          <el-table-column label="文件数" width="80">
            <template #default="{ row }">{{ row.data_count }}</template>
          </el-table-column>
          <el-table-column prop="created_at" label="创建时间" width="155" />
          <el-table-column prop="published_at" label="发布时间" width="155">
            <template #default="{ row }">{{ row.published_at || '—' }}</template>
          </el-table-column>
          <el-table-column label="操作" width="230" fixed="right">
            <template #default="{ row }">
              <el-button v-if="row.status === 'draft' && auth.canWrite" link type="success" size="small" @click="onPublish(row)">发布</el-button>
              <el-button link type="primary" size="small" @click="onDownload(row)">下载</el-button>
              <el-button link type="primary" size="small" @click="showLineage(row)">血缘</el-button>
              <el-button v-if="row.status === 'draft' && auth.canWrite" link type="danger" size="small" @click="onDelete(row)">删除</el-button>
            </template>
          </el-table-column>
          <template #empty><div class="empty-tip">暂无版本记录</div></template>
        </el-table>
        <div v-if="auth.canWrite" style="margin-top: 12px">
          <el-button size="small" @click="copyVisible = true">➕ 从现有版本新建草稿版本</el-button>
        </div>
      </el-tab-pane>

      <el-tab-pane label="血缘视图" name="lineage">
        <div v-if="lineageLoading" v-loading="true" style="height: 160px" />
        <template v-else-if="lineage">
          <!-- 链式文本展示：V1 → V2 → V4 -->
          <div class="lineage-chain">
            <template v-for="(node, i) in chainNodes" :key="node.id">
              <span v-if="i > 0" class="chain-arrow">→</span>
              <el-tooltip placement="top" :content="`${taskName(node.source_task_type)}${node.source_task_id ? ' #' + node.source_task_id : ''} · ${node.created_at || ''}${node.quality_score != null ? ' · 质量分 ' + node.quality_score : ''}`">
                <span class="chain-node" :class="{ current: node.id === lineage.current?.id, published: node.status === 'published' }"
                      @click="showLineage(node)">
                  {{ node.version }}
                  <span class="chain-stage">{{ node.stage }}</span>
                </span>
              </el-tooltip>
            </template>
          </div>
          <el-descriptions v-if="lineage.current" :column="2" border size="small" style="margin-top: 16px">
            <el-descriptions-item label="当前版本">{{ lineage.current.version }}</el-descriptions-item>
            <el-descriptions-item label="阶段">{{ lineage.current.stage }}</el-descriptions-item>
            <el-descriptions-item label="状态">{{ lineage.current.status === 'published' ? '已发布' : '草稿' }}</el-descriptions-item>
            <el-descriptions-item label="来源任务">
              {{ taskName(lineage.current.source_task_type) }}
              <span v-if="lineage.current.source_task_id">#{{ lineage.current.source_task_id }}</span>
            </el-descriptions-item>
            <el-descriptions-item label="质量分">{{ lineage.current.quality_score ?? '未评估' }}</el-descriptions-item>
            <el-descriptions-item label="创建时间">{{ lineage.current.created_at }}</el-descriptions-item>
          </el-descriptions>
          <div v-if="lineage.descendants?.length" style="margin-top: 14px">
            <div class="muted" style="margin-bottom: 8px">下游子孙版本（{{ lineage.descendants.length }}）：</div>
            <el-tag v-for="d in lineage.descendants" :key="d.id" style="margin: 0 8px 8px 0" effect="plain"
                    :type="d.status === 'published' ? 'success' : 'warning'">
              {{ d.version }} · {{ d.stage }}
            </el-tag>
          </div>
        </template>
        <el-empty v-else description="选择版本后展示血缘" :image-size="60" />
      </el-tab-pane>
    </el-tabs>

    <!-- 新建草稿版本 -->
    <el-dialog v-model="copyVisible" title="新建草稿版本" width="440px" append-to-body>
      <el-form label-width="90px">
        <el-form-item label="基准版本">
          <el-select v-model="copyForm.base_version_id" style="width: 100%">
            <el-option v-for="v in versions" :key="v.id" :value="v.id"
                       :label="`${v.version}（${v.stage} · ${v.status === 'published' ? '已发布' : '草稿'}）`" />
          </el-select>
        </el-form-item>
        <el-form-item label="版本说明"><el-input v-model="copyForm.description" type="textarea" :rows="2" /></el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="copyVisible = false">取消</el-button>
        <el-button type="primary" @click="onCopy">创建</el-button>
      </template>
    </el-dialog>
  </el-dialog>
</template>

<script setup>
import { computed, reactive, ref, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import api from '../api'
import { useAuthStore } from '../stores/auth'
import { TASK_TYPE_NAMES } from '../utils'

const props = defineProps({ dataset: Object })
const emit = defineEmits(['changed'])
const auth = useAuthStore()

const visible = ref(false)
const tab = ref('list')
const loading = ref(false)
const versions = ref([])
const lineage = ref(null)
const lineageLoading = ref(false)
const copyVisible = ref(false)
const copyForm = reactive({ base_version_id: null, description: '' })

function open() {
  visible.value = true
  tab.value = 'list'
  lineage.value = null
  load()
}
defineExpose({ open })

function taskName(t) { return TASK_TYPE_NAMES[t] || t || '—' }

// 切换到血缘视图时若未选版本，默认展示当前版本血缘
function onTabChange(name) {
  if (name === 'lineage' && !lineage.value && versions.value.length) {
    const cur = versions.value.find((v) => v.id === props.dataset?.current_version?.id) || versions.value[0]
    showLineage(cur)
  }
}
watch(tab, onTabChange)

async function load() {
  if (!props.dataset) return
  loading.value = true
  try {
    const data = await api.get(`/datasets/${props.dataset.id}/versions`)
    versions.value = data.items
    copyForm.base_version_id = props.dataset.current_version?.id ?? data.items[0]?.id
  } finally { loading.value = false }
}

// 血缘链：从 V1 到当前版本的祖先链（正序）
const chainNodes = computed(() => {
  if (!lineage.value) return []
  return [...lineage.value.ancestors].reverse().concat([lineage.value.current])
})

async function showLineage(v) {
  tab.value = 'lineage'
  lineageLoading.value = true
  try {
    lineage.value = await api.get(`/datasets/${props.dataset.id}/versions/${v.id}/lineage`)
  } finally { lineageLoading.value = false }
}

async function onPublish(v) {
  const { value } = await ElMessageBox.prompt(`确认发布 ${v.version}？发布后只读不可删除。可填写版本说明：`,
    '发布版本', { confirmButtonText: '发布', cancelButtonText: '取消', inputPlaceholder: '版本说明（可选）' })
  await api.post(`/datasets/${props.dataset.id}/versions/${v.id}/publish`, { description: value || '' })
  ElMessage.success(`${v.version} 已发布`)
  load(); emit('changed')
}

async function onDelete(v) {
  await ElMessageBox.confirm(`确认删除草稿版本 ${v.version}？版本目录将同步清理。`, '删除版本', { type: 'warning' })
  try {
    await api.delete(`/datasets/${props.dataset.id}/versions/${v.id}`)
    ElMessage.success('已删除')
    load(); emit('changed')
  } catch { /* 拦截器已 Toast 具体引用来源 */ }
}

function onDownload(v) {
  window.open(`/api/datasets/${props.dataset.id}/versions/${v.id}/download`, '_blank')
}

async function onCopy() {
  await api.post(`/datasets/${props.dataset.id}/versions`, { ...copyForm })
  ElMessage.success('已创建新草稿版本')
  copyVisible.value = false
  load(); emit('changed')
}
</script>

<style scoped>
.muted { color: #64748b; }
.lineage-chain {
  display: flex; align-items: center; flex-wrap: wrap; gap: 4px;
  background: #0f1729; border: 1px solid #24304d; border-radius: 10px; padding: 18px;
}
.chain-arrow { color: #60a5fa; margin: 0 6px; font-weight: 700; }
.chain-node {
  display: inline-flex; flex-direction: column; align-items: center;
  padding: 8px 14px; border-radius: 8px; cursor: pointer;
  border: 1px solid #24304d; background: #131c31; color: #e2e8f0; font-weight: 600;
}
.chain-node:hover { border-color: #60a5fa; }
.chain-node.current { border-color: #60a5fa; background: rgba(96, 165, 250, 0.15); color: #60a5fa; }
.chain-node.published { border-color: rgba(52, 211, 153, 0.5); }
.chain-stage { font-size: 0.68rem; color: #94a3b8; font-weight: 400; margin-top: 2px; }
.empty-tip { text-align: center; color: #64748b; padding: 30px 0; }
</style>
