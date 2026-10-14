<template>
  <div class="page">
    <div class="page-header">
      <h2><span>📊</span> 数据可视化中心</h2>
      <p>选择数据类型与数据集，基于真实文件数据生成可视化图表</p>
    </div>

    <!-- 类型选择卡片 -->
    <div class="type-grid">
      <div v-for="(m, t) in TYPE_META" :key="t" class="type-card" @click="selectType(t)">
        <div class="type-icon">{{ m.icon }}</div>
        <div class="type-name">{{ m.title.replace('获取', '') }}可视化</div>
        <div class="type-count">{{ dash.by_type?.[t]?.count ?? 0 }} 个数据集</div>
      </div>
    </div>

    <!-- 数据集列表 -->
    <div v-if="selectedType" class="table-card" style="margin-top: 16px" v-loading="loading">
      <h4 style="margin: 0 0 12px">{{ TYPE_META[selectedType].icon }} 选择{{ TYPE_META[selectedType].short }}进行可视化</h4>
      <el-table :data="rows" size="default">
        <el-table-column prop="name" label="名称" min-width="200" show-overflow-tooltip />
        <el-table-column prop="file_count" label="文件数" width="90" />
        <el-table-column label="大小" width="110"><template #default="{ row }">{{ fmtSize(row.size_bytes) }}</template></el-table-column>
        <el-table-column prop="status" label="状态" width="90" />
        <el-table-column prop="created_at" label="创建时间" width="160" />
        <el-table-column label="操作" width="120">
          <template #default="{ row }">
            <el-button type="primary" size="small" @click="$router.push(`/viz/${selectedType}/${row.id}`)">进入可视化</el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>

    <!-- 平台总览图表 -->
    <div v-if="!selectedType" style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px">
      <div class="table-card">
        <h4 style="margin: 0 0 8px">近 14 天数据接入趋势</h4>
        <EChart :option="trendOption" height="300px" />
      </div>
      <div class="table-card">
        <h4 style="margin: 0 0 8px">数据分布（密级）</h4>
        <EChart :option="distOption" height="300px" />
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import api from '../api'
import { fmtSize, TYPE_META } from '../utils'
import EChart from '../components/EChart.vue'

const selectedType = ref(null)
const rows = ref([])
const loading = ref(false)
const dash = ref({})
const trend = ref({ days: [], series: {} })
const dist = ref({})

const COLORS = { voice: '#60a5fa', radar: '#34d399', image: '#fbbf24', video: '#ef4444', text: '#a78bfa' }

async function selectType(t) {
  selectedType.value = selectedType.value === t ? null : t
  if (!selectedType.value) return
  loading.value = true
  try {
    const data = await api.get(`/${t}`, { params: { page: 1, page_size: 50 } })
    rows.value = data.items
  } finally { loading.value = false }
}

const trendOption = computed(() => ({
  tooltip: { trigger: 'axis' },
  legend: { textStyle: { color: '#94a3b8' } },
  grid: { left: 40, right: 20, top: 40, bottom: 30 },
  xAxis: { type: 'category', data: trend.value.days || [] },
  yAxis: { type: 'value', splitLine: { lineStyle: { color: '#24304d' } } },
  series: Object.entries(trend.value.series || {}).map(([t, data]) => ({
    name: TYPE_META[t].title.replace('获取', ''), type: 'line', smooth: true, data,
    lineStyle: { color: COLORS[t] }, itemStyle: { color: COLORS[t] },
  })),
}))

const distOption = computed(() => {
  const data = Object.entries(dist.value.secret_level || {}).map(([name, value]) => ({ name, value }))
  return {
    tooltip: { trigger: 'item' },
    legend: { bottom: 0, textStyle: { color: '#94a3b8' } },
    series: [{
      type: 'pie', radius: ['40%', '70%'], data,
      label: { color: '#cbd5e1' },
      itemStyle: { borderColor: '#0c1220', borderWidth: 2 },
      color: ['#ef4444', '#fbbf24', '#60a5fa', '#34d399'],
    }],
  }
})

onMounted(async () => {
  dash.value = await api.get('/dashboard/stats')
  trend.value = await api.get('/dashboard/trend')
  dist.value = await api.get('/dashboard/distribution')
})
</script>

<style scoped>
.type-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; }
.type-card {
  background: linear-gradient(135deg, #131c31, #16223c);
  border: 1px solid #24304d; border-radius: 12px;
  padding: 22px 16px; text-align: center; cursor: pointer;
  transition: border-color 0.2s, transform 0.2s;
}
.type-card:hover { border-color: #60a5fa; transform: translateY(-2px); }
.type-icon { font-size: 2rem; margin-bottom: 10px; }
.type-name { font-weight: 600; margin-bottom: 6px; }
.type-count { color: #94a3b8; font-size: 0.78rem; }
</style>
