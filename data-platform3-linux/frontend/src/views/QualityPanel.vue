<template>
  <div class="page">
    <div class="page-header">
      <h2><span>🛡️</span> 数据质量评估</h2>
      <p>文件质量检查（完整性/内容有效性/一致性/时效性/规范性/可访问性）+ 参考指标对照（不代表人工合规审查），支持版本级评估与报告导出</p>
    </div>

    <div class="filter-bar">
      <el-radio-group v-model="targetKind" @change="result = null">
        <el-radio-button value="raw">数据获取层</el-radio-button>
        <el-radio-button value="dataset">预处理数据集（版本级）</el-radio-button>
      </el-radio-group>
      <template v-if="targetKind === 'raw'">
        <el-select v-model="form.data_type" placeholder="数据类型" style="width: 120px" @change="form.data_id = null; result = null">
          <el-option v-for="(n, t) in TYPE_NAMES" :key="t" :label="n" :value="t" />
        </el-select>
        <el-select v-model="form.data_id" placeholder="选择数据" filterable style="width: 220px" @focus="loadRawOptions" @change="result = null">
          <el-option v-for="s in rawOptions" :key="s.id" :label="s.name" :value="s.id" />
        </el-select>
      </template>
      <template v-else>
        <el-select v-model="form.dataset_id" placeholder="选择数据集" filterable style="width: 220px" @change="onDatasetChange">
          <el-option v-for="d in datasets" :key="d.id" :label="`${d.name}（${TYPE_NAMES[d.data_type]}）`" :value="d.id" />
        </el-select>
        <el-select v-model="form.version_id" placeholder="评估版本（默认当前版本）" filterable style="width: 220px">
          <el-option v-for="v in datasetVersionsList" :key="v.id" :value="v.id"
                     :label="`${v.version}（${v.stage} · ${v.status === 'published' ? '已发布' : '草稿'}${v.quality_score != null ? ' · ' + v.quality_score + '分' : ''}）`" />
        </el-select>
      </template>
      <el-popover placement="bottom-start" width="420" trigger="click">
        <template #reference>
          <el-button>📋 评估标准{{ form.standard_ids.length ? `（已选 ${form.standard_ids.length} 项）` : '（默认全部现行）' }}</el-button>
        </template>
        <el-checkbox-group v-model="form.standard_ids">
          <div v-for="s in standards" :key="s.id" style="padding: 4px 0">
            <el-checkbox :value="s.id">
              {{ s.code }}《{{ s.name }}》
              <el-tag v-if="s.is_custom" size="small" type="warning" effect="plain">自定义</el-tag>
              <el-tag v-else-if="s.status !== '现行'" size="small" type="info" effect="plain">{{ s.status }}</el-tag>
            </el-checkbox>
          </div>
        </el-checkbox-group>
        <div style="margin-top: 8px; font-size: 0.78rem; color: #94a3b8">不勾选时默认评估全部「现行」标准；每项标准单独计算合规率</div>
      </el-popover>
      <el-button type="primary" :disabled="!auth.canOperate" :loading="evaluating" @click="onEvaluate">🔍 发起评估</el-button>
      <template v-if="result">
        <el-button @click="onExport('json')">导出 JSON</el-button>
        <el-button @click="onExport('csv')">导出 CSV</el-button>
      </template>
    </div>

    <div v-if="evaluating" v-loading="true" style="height: 200px" />
    <template v-else-if="result">
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px">
        <div class="table-card">
          <h4 style="margin: 0 0 8px">六维度雷达图（综合得分：<span :style="{ color: scoreColor(result.overall_score) }">{{ result.overall_score }}</span>）</h4>
          <EChart :option="radarOption" height="340px" />
        </div>
        <div>
          <div class="table-card" style="margin-bottom: 16px">
            <h4 style="margin: 0 0 10px">指标明细</h4>
            <el-descriptions :column="2" border size="small">
              <el-descriptions-item v-for="(name, key) in DIM_NAMES" :key="key" :label="name">
                <span :style="{ color: scoreColor(result[key]) }">{{ result[key] }}</span>
              </el-descriptions-item>
            </el-descriptions>
            <div style="margin-top: 10px; font-size: 0.8rem; color: #94a3b8">
              <div v-for="(c, i) in result.details?.checks || []" :key="i">· {{ c }}</div>
            </div>
          </div>
          <div class="table-card">
            <h4 style="margin: 0 0 10px">参考标准合规性</h4>
            <el-table :data="result.standards_compliance || []" size="small">
              <el-table-column type="expand">
                <template #default="{ row }">
                  <div style="padding: 8px 16px">
                    <template v-if="row.rule_results?.length">
                      <div v-for="(r, i) in row.rule_results" :key="i" style="font-size: 0.8rem; padding: 3px 0">
                        <el-tag size="small" :type="r.passed ? 'success' : 'danger'" effect="plain">{{ r.passed ? '通过' : '未通过' }}</el-tag>
                        {{ r.name }}：{{ metricLabel(r.metric) }} {{ r.op }} {{ r.threshold }}（实际 {{ r.actual }}）
                      </div>
                    </template>
                    <template v-else-if="row.dimension_scores">
                      <span v-for="(v, k) in row.dimension_scores" :key="k" style="font-size: 0.8rem; margin-right: 14px">
                        {{ DIM_NAMES[k] || k }} <b :style="{ color: scoreColor(v) }">{{ v }}</b>
                      </span>
                    </template>
                    <span v-else class="muted" style="font-size: 0.8rem">无明细</span>
                  </div>
                </template>
              </el-table-column>
              <el-table-column prop="code" label="标准编号" width="170" />
              <el-table-column label="合规率" width="140">
                <template #default="{ row }">
                  <el-progress :percentage="row.compliance_rate" :stroke-width="10"
                               :color="row.compliance_rate >= 80 ? '#34d399' : row.compliance_rate >= 60 ? '#fbbf24' : '#ef4444'" />
                </template>
              </el-table-column>
              <el-table-column label="结论" width="90">
                <template #default="{ row }">
                  <el-tag size="small" effect="dark"
                          :type="row.level === '符合' ? 'success' : row.level === '基本符合' ? 'warning' : 'danger'">{{ row.level }}</el-tag>
                </template>
              </el-table-column>
            </el-table>
          </div>
        </div>
      </div>
    </template>
    <el-empty v-else description="请选择评估对象并发起评估" />

    <!-- 历史评估记录 -->
    <div v-if="history.length" class="table-card" style="margin-top: 16px">
      <h4 style="margin: 0 0 10px">历史评估记录</h4>
      <el-table :data="history" size="small">
        <el-table-column prop="created_at" label="评估时间" width="160" />
        <el-table-column label="综合得分" width="100">
          <template #default="{ row }"><span :style="{ color: scoreColor(row.overall_score) }">{{ row.overall_score }}</span></template>
        </el-table-column>
        <el-table-column prop="completeness" label="完整性" width="80" />
        <el-table-column prop="accuracy" label="内容有效性" width="80" />
        <el-table-column prop="consistency" label="一致性" width="80" />
        <el-table-column prop="timeliness" label="时效性" width="80" />
      </el-table>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import api from '../api'
import { useAuthStore } from '../stores/auth'
import { TYPE_NAMES, QUALITY_METRICS } from '../utils'
import EChart from '../components/EChart.vue'

const DIM_NAMES = { completeness: '完整性', accuracy: '内容有效性', consistency: '一致性',
                    timeliness: '时效性', standardization: '规范性', accessibility: '可访问性' }

const route = useRoute()
const auth = useAuthStore()
const targetKind = ref('raw')
const datasets = ref([])
const rawOptions = ref([])
const datasetVersionsList = ref([])
const evaluating = ref(false)
const result = ref(null)
const history = ref([])
const form = reactive({ data_type: 'voice', data_id: null, dataset_id: null, version_id: null, standard_ids: [] })
const standards = ref([])

function metricLabel(key) {
  const m = QUALITY_METRICS.find((x) => x.key === key)
  return m ? m.name : key
}

function scoreColor(s) { return s >= 80 ? '#34d399' : s >= 60 ? '#fbbf24' : '#ef4444' }

async function loadRawOptions() {
  const data = await api.get(`/${form.data_type}`, { params: { page: 1, page_size: 100 } })
  rawOptions.value = data.items
}

async function onDatasetChange() {
  result.value = null
  const ds = datasets.value.find((d) => d.id === form.dataset_id)
  const vs = await api.get(`/datasets/${form.dataset_id}/versions`)
  datasetVersionsList.value = [...vs.items].sort((a, b) => a.version_no - b.version_no)
  form.version_id = ds?.current_version?.id ?? datasetVersionsList.value.at(-1)?.id
}

async function onEvaluate() {
  evaluating.value = true
  try {
    if (targetKind.value === 'raw') {
      if (!form.data_id) { ElMessage.error('请选择数据'); return }
      result.value = await api.post('/quality/evaluate',
        { data_type: form.data_type, data_id: form.data_id, standard_ids: form.standard_ids })
      const h = await api.get(`/quality/${form.data_type}/${form.data_id}`)
      history.value = h.items
    } else {
      if (!form.dataset_id || !form.version_id) { ElMessage.error('请选择数据集与版本'); return }
      const ds = datasets.value.find((d) => d.id === form.dataset_id)
      result.value = await api.post('/quality/evaluate',
        { data_type: ds.data_type, data_id: form.dataset_id, version_id: form.version_id })
      const h = await api.get(`/quality/${ds.data_type}/${form.dataset_id}`, { params: { version_id: form.version_id } })
      history.value = h.items
    }
    ElMessage.success(`评估完成，综合得分 ${result.value.overall_score}`)
  } catch { /* 已提示 */ } finally { evaluating.value = false }
}

function onExport(fmt) {
  const dataType = targetKind.value === 'raw' ? form.data_type : datasets.value.find((d) => d.id === form.dataset_id)?.data_type
  const dataId = targetKind.value === 'raw' ? form.data_id : form.dataset_id
  const vid = targetKind.value === 'dataset' && form.version_id ? `&version_id=${form.version_id}` : ''
  window.open(`/api/quality/${dataType}/${dataId}/export?fmt=${fmt}${vid}`, '_blank')
}

const radarOption = computed(() => ({
  radar: {
    indicator: Object.values(DIM_NAMES).map((name) => ({ name, max: 100 })),
    axisName: { color: '#94a3b8' },
    splitLine: { lineStyle: { color: '#24304d' } },
    splitArea: { areaStyle: { color: ['transparent'] } },
  },
  series: [{
    type: 'radar',
    data: [{
      value: Object.keys(DIM_NAMES).map((k) => result.value?.[k] ?? 0),
      name: '质量得分',
      areaStyle: { color: 'rgba(96,165,250,0.3)' },
      lineStyle: { color: '#60a5fa' },
    }],
  }],
}))

onMounted(async () => {
  const data = await api.get('/datasets', { params: { page: 1, page_size: 100 } })
  datasets.value = data.items
  const sd = await api.get('/standards')
  standards.value = sd.items
  if (route.query.dataset_id) {
    targetKind.value = 'dataset'
    form.dataset_id = Number(route.query.dataset_id)
    await onDatasetChange()
    const requested=Number(route.query.version_id)
    if(requested&&datasetVersionsList.value.some(v=>v.id===requested))form.version_id=requested
  }
})
</script>
