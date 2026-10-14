<template>
  <div class="page">
    <div class="page-header">
      <h2><span>🗂️</span> 数据集管理</h2>
      <p>预处理数据集全生命周期管理：版本化（V1…Vn）、草稿/发布状态机、血缘追溯与质量评估挂载</p>
    </div>

    <div class="stat-cards">
      <div class="stat-card"><div class="stat-label">数据集总数</div><div class="stat-value">{{ stats.dataset_count ?? '-' }}</div></div>
      <div class="stat-card"><div class="stat-label">版本总数</div><div class="stat-value success">{{ stats.version_count ?? '-' }}</div></div>
      <div class="stat-card"><div class="stat-label">接入数据总量</div><div class="stat-value warning">{{ totalRaw }}</div></div>
    </div>

    <div class="filter-bar">
      <el-input v-model="query.search" placeholder="搜索数据集名称…" clearable style="width: 200px" @change="load(1)" />
      <el-select v-model="query.data_type" placeholder="数据类型" clearable style="width: 130px" @change="load(1)">
        <el-option v-for="(n, t) in TYPE_NAMES" :key="t" :label="n" :value="t" />
      </el-select>
      <el-select v-model="query.stage" placeholder="阶段（当前版本）" clearable style="width: 150px" @change="load(1)">
        <el-option v-for="s in STAGES" :key="s" :label="s" :value="s" />
      </el-select>
      <div style="margin-left: auto">
        <el-button v-if="auth.canWrite" type="primary" @click="openCreate">➕ 新建数据集</el-button>
      </div>
    </div>

    <div class="table-card" v-loading="loading">
      <el-table :data="rows" @sort-change="onSort">
        <el-table-column prop="name" label="数据集名称" min-width="170" sortable="custom" show-overflow-tooltip />
        <el-table-column label="类型" width="80">
          <template #default="{ row }">{{ TYPE_NAMES[row.data_type] }}</template>
        </el-table-column>
        <el-table-column label="密级" width="76">
          <template #default="{ row }">
            <el-tag size="small" effect="dark" :type="SECRET_TAGS[row.secret_level] || 'info'">{{ row.secret_level || '内部' }}</el-tag>
          </template>
        </el-table-column>
        <!-- 当前版本列（8.7）：tag 显示 V{n}，已发布 success / 草稿 warning -->
        <el-table-column label="当前版本" width="140">
          <template #default="{ row }">
            <el-tag v-if="row.current_version" effect="dark"
                    :type="row.current_version.status === 'published' ? 'success' : 'warning'">
              {{ row.current_version.version }} · {{ row.current_version.status === 'published' ? '已发布' : '草稿' }}
            </el-tag>
            <span v-else class="muted">—</span>
          </template>
        </el-table-column>
        <el-table-column label="阶段" width="90">
          <template #default="{ row }">{{ row.current_version?.stage || '—' }}</template>
        </el-table-column>
        <el-table-column label="版本数" width="80">
          <template #default="{ row }">{{ row.version_count }}</template>
        </el-table-column>
        <el-table-column label="质量分" width="90">
          <template #default="{ row }">
            <span v-if="row.current_version?.quality_score != null"
                  :style="{ color: row.current_version.quality_score >= 80 ? '#34d399' : '#fbbf24' }">
              {{ row.current_version.quality_score }}
            </span>
            <span v-else class="muted">未评估</span>
          </template>
        </el-table-column>
        <el-table-column prop="data_count" label="文件数" width="80" sortable="custom" />
        <el-table-column prop="created_at" label="创建时间" width="160" sortable="custom" />
        <el-table-column label="操作" width="260" fixed="right">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="openVersions(row)">版本历史</el-button>
            <el-button link type="primary" size="small" @click="openDownload(row)">下载</el-button>
            <el-button link type="primary" size="small" @click="goQuality(row)">质量评估</el-button>
            <el-button v-if="auth.canWrite" link type="danger" size="small" @click="onDelete(row)">删除</el-button>
          </template>
        </el-table-column>
        <template #empty><div class="empty-tip">暂无数据集，点击"新建数据集"创建（自动生成 V1 初始版本）</div></template>
      </el-table>
      <div class="pager">
        <el-pagination background layout="total, sizes, prev, pager, next" :total="total"
                       v-model:current-page="query.page" v-model:page-size="query.page_size"
                       :page-sizes="[10, 20, 50, 100]" @change="load()" />
      </div>
    </div>

    <!-- 新建数据集 -->
    <el-dialog v-model="createVisible" title="新建预处理数据集" width="540px">
      <el-alert type="info" :closable="false" style="margin-bottom: 12px"
                title="创建后自动生成 V1 初始版本（草稿），并从所选来源数据复制文件" />
      <el-form label-width="90px">
        <el-form-item label="名称" required><el-input v-model="form.name" /></el-form-item>
        <el-form-item label="数据类型" required>
          <el-select v-model="form.data_type" filterable style="width: 100%" @change="form.source_data_ids = []; loadSources()">
            <el-option v-for="(n, t) in TYPE_NAMES" :key="t" :label="n" :value="t" />
          </el-select>
        </el-form-item>
        <el-form-item label="来源数据">
          <el-select v-model="form.source_data_ids" multiple filterable style="width: 100%" placeholder="选择数据获取层数据（可选）"
                     @focus="loadSources">
            <el-option v-for="s in sources" :key="s.id" :label="`${s.name}（${s.file_count} 个文件）`" :value="s.id" />
          </el-select>
        </el-form-item>
        <el-form-item label="密级" required>
          <el-radio-group v-model="form.secret_level">
            <el-radio-button v-for="lv in SECRET_LEVELS" :key="lv" :value="lv">{{ lv }}</el-radio-button>
          </el-radio-group>
          <div class="muted" style="font-size: 0.78rem; margin-top: 4px">访客仅可见/可操作获授权密级及以下的数据集</div>
        </el-form-item>
        <el-form-item label="描述"><el-input v-model="form.description" type="textarea" :rows="2" /></el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="createVisible = false">取消</el-button>
        <el-button type="primary" :loading="saving" @click="onCreate">创建</el-button>
      </template>
    </el-dialog>

    <!-- 下载对话框：选择版本 + 内容（数据/标注/清单） -->
    <el-dialog v-model="dlVisible" :title="`下载数据集「${dlDataset?.name || ''}」`" width="480px">
      <el-form label-width="90px">
        <el-form-item label="选择版本" required>
          <el-select v-model="dlVersionId" filterable style="width: 100%">
            <el-option v-for="v in dlVersions" :key="v.id" :value="v.id"
                       :label="`${v.version}（${v.stage} · ${v.status === 'published' ? '已发布' : '草稿'} · ${v.file_count ?? '-'} 个文件）`" />
          </el-select>
        </el-form-item>
        <el-form-item label="下载内容">
          <el-checkbox-group v-model="dlInclude">
            <el-checkbox value="data">数据文件</el-checkbox>
            <el-checkbox value="labels">标注文件</el-checkbox>
            <el-checkbox value="manifest">清单(manifest)</el-checkbox>
          </el-checkbox-group>
          <div class="muted" style="font-size: 0.78rem">全不勾选 = 打包全部内容（ZIP）</div>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dlVisible = false">取消</el-button>
        <el-button type="primary" :disabled="!dlVersionId" @click="doDownload">⬇️ 下载 ZIP</el-button>
      </template>
    </el-dialog>

    <VersionHistoryDialog ref="versionDialog" :dataset="currentDataset" @changed="load()" />
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import api from '../api'
import { useAuthStore } from '../stores/auth'
import { fmtSize, STAGES, TYPE_NAMES, SECRET_LEVELS, SECRET_TAGS } from '../utils'
import VersionHistoryDialog from '../components/VersionHistoryDialog.vue'

const router = useRouter()
const auth = useAuthStore()
const rows = ref([])
const total = ref(0)
const loading = ref(false)
const stats = ref({})
const query = reactive({ page: 1, page_size: 20, search: '', data_type: '', stage: '', sort: '' })

const totalRaw = computed(() => {
  const byType = stats.value.by_type || {}
  return fmtSize(Object.values(byType).reduce((s, t) => s + (t.size || 0), 0))
})

async function load(page) {
  if (page) query.page = page
  loading.value = true
  try {
    const data = await api.get('/datasets', { params: query })
    rows.value = data.items
    total.value = data.total
  } finally { loading.value = false }
}
function onSort({ prop, order }) {
  query.sort = order ? (order === 'descending' ? '-' : '') + prop : ''
  load()
}

// ---- 新建 ----
const createVisible = ref(false)
const saving = ref(false)
const sources = ref([])
const form = reactive({ name: '', data_type: 'voice', description: '', source_data_ids: [], secret_level: '内部' })
function openCreate() { createVisible.value = true; loadSources() }
async function loadSources() {
  const data = await api.get(`/${form.data_type}`, { params: { page: 1, page_size: 100 } })
  sources.value = data.items
}
async function onCreate() {
  if (!form.name) return ElMessage.error('名称不能为空')
  saving.value = true
  try {
    await api.post('/datasets', form)
    ElMessage.success('数据集已创建（含 V1 初始版本）')
    createVisible.value = false
    load()
  } catch { /* 已提示 */ } finally { saving.value = false }
}

// ---- 版本历史 ----
const versionDialog = ref(null)
const currentDataset = ref(null)
function openVersions(row) {
  currentDataset.value = row
  versionDialog.value.open()
}

// ---- 下载（版本 + 内容选择） ----
const dlVisible = ref(false)
const dlDataset = ref(null)
const dlVersions = ref([])
const dlVersionId = ref(null)
const dlInclude = ref([])
async function openDownload(row) {
  dlDataset.value = row
  dlInclude.value = []
  const data = await api.get(`/datasets/${row.id}/versions`)
  dlVersions.value = [...data.items].sort((a, b) => a.version_no - b.version_no)
  dlVersionId.value = row.current_version?.id ?? dlVersions.value.at(-1)?.id ?? null
  dlVisible.value = true
}
function doDownload() {
  const include = dlInclude.value.length ? `?include=${dlInclude.value.join(',')}` : ''
  window.open(`/api/datasets/${dlDataset.value.id}/versions/${dlVersionId.value}/download${include}`, '_blank')
  dlVisible.value = false
}
function goQuality(row) {
  router.push({ path: '/quality', query: { dataset_id: row.id } })
}
async function onDelete(row) {
  await ElMessageBox.confirm(`确认删除数据集「${row.name}」？（软删除，文件保留）`, '删除确认', { type: 'warning' })
  await api.delete(`/datasets/${row.id}`)
  ElMessage.success('已删除')
  load()
}

onMounted(async () => {
  load()
  stats.value = await api.get('/dashboard/stats')
})
</script>

<style scoped>
.muted { color: #64748b; }
</style>
