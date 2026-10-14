<template>
  <div class="page">
    <div class="page-header">
      <h2><span>📤</span> 生成结果回写</h2>
      <p>模型生成结果独立回写存储（writeback_results），支持文件上传与下载</p>
    </div>

    <div class="filter-bar">
      <div style="margin-left: auto">
        <el-button v-if="auth.canWrite" type="primary" @click="openCreate">➕ 新建回写</el-button>
      </div>
    </div>

    <div class="table-card" v-loading="loading">
      <el-table :data="rows" @sort-change="onSort">
        <el-table-column prop="model_name" label="模型名称" min-width="150" sortable="custom" />
        <el-table-column prop="result_type" label="结果类型" width="120" />
        <el-table-column label="关联数据" width="140">
          <template #default="{ row }">
            {{ row.original_data_type ? `${TYPE_NAMES[row.original_data_type] || row.original_data_type} #${row.original_data_id}` : '—' }}
          </template>
        </el-table-column>
        <el-table-column prop="version" label="版本" width="100" />
        <el-table-column label="大小" width="110"><template #default="{ row }">{{ fmtSize(row.size_bytes) }}</template></el-table-column>
        <el-table-column prop="status" label="状态" width="90">
          <template #default="{ row }">
            <el-tag size="small" effect="dark" :type="row.status === '成功' ? 'success' : 'danger'">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="created_at" label="回写时间" width="160" sortable="custom" />
        <el-table-column label="操作" width="100">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="onDownload(row)">下载</el-button>
          </template>
        </el-table-column>
        <template #empty><div class="empty-tip">暂无回写记录</div></template>
      </el-table>
      <div class="pager">
        <el-pagination background layout="total, sizes, prev, pager, next" :total="total"
                       v-model:current-page="page" v-model:page-size="pageSize"
                       :page-sizes="[10, 20, 50, 100]" @change="load()" />
      </div>
    </div>

    <el-dialog v-model="createVisible" title="新建结果回写" width="520px">
      <el-form label-width="90px">
        <el-form-item label="模型名称" required><el-input v-model="form.model_name" /></el-form-item>
        <el-form-item label="结果类型"><el-input v-model="form.result_type" placeholder="如：语音分割 / 雷达目标" /></el-form-item>
        <el-form-item label="关联数据">
          <el-select v-model="form.original_data_type" placeholder="类型" style="width: 120px" @change="form.original_data_id = null; loadSourceOptions()">
            <el-option v-for="(n, t) in TYPE_NAMES" :key="t" :label="n" :value="t" />
          </el-select>
          <el-select v-model="form.original_data_id" placeholder="数据" style="width: 220px; margin-left: 8px" @focus="loadSourceOptions">
            <el-option v-for="s in sourceOptions" :key="s.id" :label="s.name" :value="s.id" />
          </el-select>
        </el-form-item>
        <el-form-item label="版本号"><el-input v-model="form.version" placeholder="如：v1.0.0" /></el-form-item>
        <el-form-item label="结果文件">
          <el-upload :auto-upload="false" :show-file-list="false" :on-change="(f) => (file = f.raw)">
            <el-button>选择文件</el-button>
          </el-upload>
          <span v-if="file" style="margin-left: 10px; font-size: 0.82rem">{{ file.name }}</span>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="createVisible = false">取消</el-button>
        <el-button type="primary" :loading="saving" @click="onSave">提交回写</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'
import { useAuthStore } from '../stores/auth'
import { fmtSize, TYPE_NAMES } from '../utils'

const auth = useAuthStore()
const rows = ref([])
const total = ref(0)
const loading = ref(false)
const page = ref(1)
const pageSize = ref(20)
const sort = ref('')

async function load() {
  loading.value = true
  try {
    const data = await api.get('/writeback', { params: { page: page.value, page_size: pageSize.value, sort: sort.value } })
    rows.value = data.items
    total.value = data.total
  } finally { loading.value = false }
}
function onSort({ prop, order }) {
  sort.value = order ? (order === 'descending' ? '-' : '') + prop : ''
  load()
}

const createVisible = ref(false)
const saving = ref(false)
const file = ref(null)
const sourceOptions = ref([])
const form = reactive({ model_name: '', result_type: '', original_data_type: '', original_data_id: null, version: '' })

function openCreate() { createVisible.value = true }
async function loadSourceOptions() {
  if (!form.original_data_type) return
  const data = await api.get(`/${form.original_data_type}`, { params: { page: 1, page_size: 100 } })
  sourceOptions.value = data.items
}

async function onSave() {
  if (!form.model_name) return ElMessage.error('模型名称不能为空')
  saving.value = true
  try {
    const fd = new FormData()
    Object.entries(form).forEach(([k, v]) => { if (v != null && v !== '') fd.append(k, v) })
    if (file.value) fd.append('file', file.value)
    await api.post('/writeback', fd, { headers: { 'Content-Type': 'multipart/form-data' } })
    ElMessage.success('回写成功')
    createVisible.value = false
    load()
  } catch { /* 已提示 */ } finally { saving.value = false }
}

function onDownload(row) {
  window.open(`/api/writeback/${row.id}/download`, '_blank')
}

onMounted(load)
</script>
