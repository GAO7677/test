<template>
  <div class="page">
    <div class="page-header">
      <h2><span>📜</span> 操作日志</h2>
      <p>所有数据操作强制记录，不可删除，仅支持查询与导出 CSV</p>
    </div>

    <div class="filter-bar">
      <el-select v-model="query.module" placeholder="模块" clearable style="width: 160px" @change="load(1)">
        <el-option v-for="m in modules" :key="m.value" :label="m.label" :value="m.value" />
      </el-select>
      <el-input v-model="query.action" placeholder="操作类型" clearable style="width: 150px" @change="load(1)" />
      <el-input v-model="query.username" placeholder="用户名" clearable style="width: 150px" @change="load(1)" />
      <div style="margin-left: auto">
        <el-button @click="onExport">📥 导出 CSV</el-button>
      </div>
    </div>

    <div class="table-card" v-loading="loading">
      <el-table :data="rows" @sort-change="onSort">
        <el-table-column prop="created_at" label="时间" width="165" sortable="custom" />
        <el-table-column prop="username" label="用户" width="100" />
        <el-table-column prop="action" label="操作" width="130" />
        <el-table-column prop="module" label="模块" width="130" />
        <el-table-column prop="detail" label="详情" min-width="300" show-overflow-tooltip />
        <el-table-column prop="ip_address" label="IP" width="120" />
        <template #empty><div class="empty-tip">暂无日志</div></template>
      </el-table>
      <div class="pager">
        <el-pagination background layout="total, sizes, prev, pager, next" :total="total"
                       v-model:current-page="query.page" v-model:page-size="query.page_size"
                       :page-sizes="[10, 20, 50, 100]" @change="load()" />
      </div>
    </div>
  </div>
</template>

<script setup>
import { onMounted, reactive, ref } from 'vue'
import api from '../api'

const modules = [
  { label: '认证', value: 'auth' }, { label: '用户管理', value: 'user' },
  { label: '系统配置', value: 'settings' }, { label: '数据集', value: 'dataset' },
  { label: '数据集版本', value: 'dataset_version' }, { label: '语音数据', value: 'data_voice' },
  { label: '雷达数据', value: 'data_radar' }, { label: '图像数据', value: 'data_image' },
  { label: '视频数据', value: 'data_video' }, { label: '文本数据', value: 'data_text' },
  { label: '自动标注', value: 'auto_label' }, { label: '数据清洗', value: 'clean' },
  { label: '数据集成', value: 'integrate' }, { label: '数据转换', value: 'transform' },
  { label: '数据采样', value: 'sample' }, { label: '工作流', value: 'workflow' },
  { label: '质量评估', value: 'quality' }, { label: '语义管理', value: 'semantic' },
  { label: '结果回写', value: 'writeback' }, { label: '操作日志', value: 'logs' },
]

const rows = ref([])
const total = ref(0)
const loading = ref(false)
const query = reactive({ page: 1, page_size: 20, module: '', action: '', username: '', sort: '' })

async function load(p) {
  if (p) query.page = p
  loading.value = true
  try {
    const data = await api.get('/logs', { params: query })
    rows.value = data.items
    total.value = data.total
  } finally { loading.value = false }
}
function onSort({ prop, order }) {
  query.sort = order ? (order === 'descending' ? '-' : '') + prop : ''
  load()
}
function onExport() {
  const params = new URLSearchParams()
  if (query.module) params.set('module', query.module)
  if (query.username) params.set('username', query.username)
  window.open(`/api/logs/export?${params}`, '_blank')
}

onMounted(() => load())
</script>
