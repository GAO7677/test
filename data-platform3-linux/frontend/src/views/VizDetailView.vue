<template>
  <div class="page">
    <div class="breadcrumb">
      <router-link to="/viz-center" style="color: #94a3b8">数据可视化中心</router-link>
      <span class="sep">/</span>
      <span class="cur">{{ TYPE_META[type]?.title.replace('获取', '') }}可视化 #{{ id }}</span>
    </div>
    <div class="page-header">
      <h2><span>{{ TYPE_META[type]?.icon }}</span> {{ TYPE_META[type]?.title.replace('获取', '') }}数据可视化</h2>
      <p v-if="activeTab === 'overview'">
        按整个数据集汇总标签：共 {{ datasetFileCount }} 个文件，识别 {{ labelStats.labels.length }} 类标签
      </p>
      <p v-else-if="viz">当前文件：{{ viz.file }}</p>
    </div>

    <el-tabs v-model="activeTab" style="margin-top: 4px">
      <el-tab-pane label="数据集总览" name="overview">
        <div v-loading="statsLoading" style="min-height: 220px">
          <template v-if="labelStats.labels.length">
            <div class="overview-summary">
              <div class="summary-item"><span>数据集文件</span><strong>{{ datasetFileCount }}</strong></div>
              <div class="summary-item"><span>标签类别</span><strong>{{ labelStats.labels.length }}</strong></div>
              <div class="summary-item"><span>已标注文件</span><strong>{{ labeledFileCount }}</strong></div>
            </div>
            <div class="viz-grid">
              <div class="table-card overview-wide">
                <h4>整个数据集的标签分布</h4>
                <EChart :option="labelDistOption" height="320px" />
              </div>
              <div class="table-card">
                <h4>数据集标签清单</h4>
                <div class="dataset-label-list">
                  <el-tag v-for="item in labelStats.labels" :key="item.label" effect="plain" size="large">
                    {{ item.label }}<span class="label-count">{{ item.count }}</span>
                  </el-tag>
                </div>
              </div>
              <div class="table-card"><h4>标注源构成</h4><EChart :option="sourceOption" height="320px" /></div>
            </div>
          </template>
          <el-empty v-else-if="!statsLoading" description="该数据集暂无标注，可先在自动/手动标注中生成" :image-size="80" />
        </div>
      </el-tab-pane>
      <el-tab-pane label="单文件视图" name="single">
    <!-- 文件切换（所有模态） -->
    <div v-if="viz?.files?.length" class="preview-bar" style="margin-bottom: 14px">
      <el-button size="small" :disabled="vizIdx <= 0" @click="switchFile(vizIdx - 1)">← 上一个</el-button>
      <el-select :model-value="viz.file_id" size="small" filterable style="width: 340px" @change="loadViz">
        <el-option v-for="f in viz.files" :key="f.id" :label="f.name" :value="f.id" />
      </el-select>
      <el-button size="small" :disabled="vizIdx >= viz.files.length - 1" @click="switchFile(vizIdx + 1)">下一个 →</el-button>
      <el-checkbox v-model="showLabel" style="margin-left: auto" @change="loadImageLabel">显示标注</el-checkbox>
    </div>

    <div v-if="loading" v-loading="true" style="height: 300px" />
    <el-empty v-else-if="!viz" description="暂无可视化数据（请先上传文件）" />

    <template v-else>
      <!-- 语音：波形图、频谱图、语谱图、音高曲线 -->
      <template v-if="type === 'voice'">
        <div class="viz-grid">
          <div class="table-card"><h4>波形图</h4><EChart :option="waveOption" height="240px" /></div>
          <div class="table-card"><h4>语谱图</h4><EChart :option="spectrogramOption" height="280px" /></div>
        </div>
      </template>

      <!-- 雷达：BEV俯视图、距离-多普勒图、点云分布、目标轨迹 -->
      <template v-else-if="type === 'radar'">
        <div class="viz-grid">
          <div class="table-card"><h4>BEV 俯视图</h4><EChart :option="bevOption" height="300px" /></div>
          <div class="table-card"><h4>距离与 Z 坐标分布（非多普勒）</h4><EChart :option="rdOption" height="300px" /></div>
          <div class="table-card"><h4>点云距离分布</h4><EChart :option="distOption" height="260px" /></div>
          <div class="table-card"><h4>按距离排序的点（非跟踪轨迹）</h4><EChart :option="trajOption" height="260px" /></div>
        </div>
      </template>

      <!-- 图像：RGB直方图、色彩热力图、分辨率分布、缩略图 -->
      <template v-else-if="type === 'image'">
        <div class="viz-grid">
          <div class="table-card">
            <h4>图像预览（{{ viz.data.resolution }}）<span v-if="showLabel && imageLabel" style="font-weight: normal; font-size: 0.78rem; color: var(--text-secondary)"> · 标注来源：{{ imageLabelSource || '原始标注' }}</span></h4>
            <PreviewPane v-if="showLabel && imagePreview" type="image" :data="imagePreview" :label="imageLabel" />
            <div v-else style="text-align: center"><img :src="viz.data.thumbnail" style="max-width: 100%; border-radius: 8px" /></div>
          </div>
        </div>
      </template>

      <!-- 视频：帧序列条、时间轴、运动向量、场景切换检测 -->
      <template v-else-if="type === 'video'">
        <div class="table-card" style="margin-bottom: 16px">
          <h4>视频预览</h4>
          <VideoPreview v-if="activeTab === 'single'" :src="`/api/video/${id}/files/${viz.file_id}/download`"
                        :poster="viz.data.frames?.[0]?.image" />
        </div>
        <div class="table-card" style="margin-bottom: 16px">
          <h4>帧序列条</h4>
          <div class="frame-strip">
            <div v-for="f in viz.data.frames" :key="f.frame" class="frame-item">
              <img :src="f.image" />
              <span>{{ f.time }}s</span>
            </div>
          </div>
        </div>
        <div class="viz-grid">
          <div class="table-card"><h4>运动量时间轴</h4><EChart :option="motionOption" height="260px" /></div>
          <div class="table-card">
            <h4>场景切换检测</h4>
            <div v-if="viz.data.scene_changes?.length" style="padding: 10px">
              <el-tag v-for="t in viz.data.scene_changes" :key="t" effect="dark" type="warning" style="margin: 0 8px 8px 0">{{ t }}s</el-tag>
            </div>
            <el-empty v-else description="未检测到明显场景切换" :image-size="60" />
          </div>
        </div>
      </template>

      <!-- 文本：长度分布、词频统计TOP20 -->
      <template v-else-if="type === 'text'">
        <div class="viz-grid">
          <div class="table-card"><h4>文本长度分布</h4><EChart :option="lenOption" height="280px" /></div>
          <div class="table-card"><h4>词频统计 TOP20</h4><EChart :option="wordOption" height="280px" /></div>
        </div>
      </template>

      <!-- 标注叠加：自动标注 / 手动标注结果（语音·雷达·文本·视频·图像通用） -->
      <div v-if="showLabel" class="table-card" style="margin-top: 16px">
        <h4>标注叠加
          <span style="font-weight: normal; font-size: 0.78rem; color: var(--text-secondary)">
            · 来源：{{ imageLabelSource || '未找到标注' }}</span>
        </h4>
        <div v-if="labelSummary.length" class="viz-grid" style="margin-top: 8px">
          <div v-for="grp in labelSummary" :key="grp.name" class="table-card">
            <h5 style="margin: 0 0 6px">{{ grp.name }}（{{ grp.items.length }}）</h5>
            <el-tag v-for="(it, i) in grp.items.slice(0, 40)" :key="i" size="small"
                    style="margin: 0 6px 6px 0">{{ it }}</el-tag>
          </div>
        </div>
        <el-empty v-else description="该文件暂无标注，可在自动标注 / 手动标注中生成" :image-size="60" />
      </div>
    </template>
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import api from '../api'
import { TYPE_META } from '../utils'
import EChart from '../components/EChart.vue'
import PreviewPane from '../components/PreviewPane.vue'
import VideoPreview from '../components/VideoPreview.vue'

const props = defineProps({ type: String, id: [String, Number] })
const loading = ref(true)
const viz = ref(null)

const axisStyle = { splitLine: { lineStyle: { color: '#24304d' } } }
const grid = { left: 50, right: 20, top: 30, bottom: 30 }

const d = computed(() => viz.value?.data || {})

const waveOption = computed(() => ({
  grid, xAxis: { type: 'category', show: false, data: (d.value.waveform || []).map((_, i) => i) },
  yAxis: { type: 'value', ...axisStyle },
  series: [{ type: 'line', showSymbol: false, data: (d.value.waveform || []).map((p) => p[1]),
             lineStyle: { color: '#60a5fa', width: 1 }, areaStyle: { color: 'rgba(96,165,250,0.2)' } }],
}))
const spectrogramOption = computed(() => {
  const spec = d.value.spectrogram || []
  const data = []
  spec.forEach((row, x) => row.forEach((v, y) => data.push([x, y, v])))
  return {
    grid: { ...grid, bottom: 50 },
    xAxis: { type: 'category', name: '帧', data: spec.map((_, i) => i), show: false },
    yAxis: { type: 'category', name: '频bin', data: spec[0]?.map((_, i) => i) || [], show: false },
    visualMap: { min: -60, max: 0, calculable: true, orient: 'horizontal', left: 'center', bottom: 0,
                 inRange: { color: ['#0c1220', '#1e3a8a', '#60a5fa', '#fbbf24'] }, textStyle: { color: '#94a3b8' } },
    series: [{ type: 'heatmap', data }],
  }
})

const bevOption = computed(() => ({
  grid, xAxis: { type: 'value', name: 'X(m)', ...axisStyle }, yAxis: { type: 'value', name: 'Y(m)', ...axisStyle },
  visualMap: { min: 0, max: Math.max(...(d.value.bev || []).map((p) => p[2]), 1), show: false,
               inRange: { color: ['#1e3a8a', '#60a5fa', '#fbbf24'] } },
  series: [{ type: 'scatter', symbolSize: 4, data: (d.value.bev || []).map((p) => [p[0], p[1], p[2]]) }],
}))
const rdOption = computed(() => ({
  grid, xAxis: { type: 'value', name: '距离(m)', ...axisStyle }, yAxis: { type: 'value', name: 'Z 坐标', ...axisStyle },
  series: [{ type: 'scatter', symbolSize: 4, itemStyle: { color: '#34d399', opacity: 0.7 },
             data: (d.value.range_doppler || []).map((p) => [p[0], p[1]]) }],
}))
const distOption = computed(() => ({
  grid, xAxis: { type: 'category', data: (d.value.distribution || []).map((x) => x.range) },
  yAxis: { type: 'value', ...axisStyle },
  series: [{ type: 'bar', data: (d.value.distribution || []).map((x) => x.count), itemStyle: { color: '#60a5fa' } }],
}))
const trajOption = computed(() => ({
  grid, xAxis: { type: 'value', name: 'X(m)', ...axisStyle }, yAxis: { type: 'value', name: 'Y(m)', ...axisStyle },
  series: [{ type: 'line', data: d.value.trajectory || [], lineStyle: { color: '#ef4444' },
             itemStyle: { color: '#ef4444' }, symbolSize: 4 }],
}))


const motionOption = computed(() => ({
  grid, xAxis: { type: 'value', name: '时间(s)', ...axisStyle }, yAxis: { type: 'value', name: '帧差', ...axisStyle },
  series: [{ type: 'line', showSymbol: false, data: d.value.motion || [],
             lineStyle: { color: '#60a5fa' }, areaStyle: { color: 'rgba(96,165,250,0.15)' } }],
}))

const lenOption = computed(() => ({
  grid, xAxis: { type: 'category', data: (d.value.length_distribution || []).map((x) => x.range) },
  yAxis: { type: 'value', ...axisStyle },
  series: [{ type: 'bar', data: (d.value.length_distribution || []).map((x) => x.count), itemStyle: { color: '#60a5fa' } }],
}))
const wordOption = computed(() => ({
  grid: { ...grid, left: 110 },
  xAxis: { type: 'value', ...axisStyle },
  yAxis: { type: 'category', data: (d.value.top_words || []).map((x) => x.word).reverse() },
  series: [{ type: 'bar', data: (d.value.top_words || []).map((x) => x.count).reverse(),
             itemStyle: { color: '#34d399' } }],
}))

const showLabel = ref(false)
const imagePreview = ref(null)   // /preview 返回的 data（含 width/height/thumbnail）
const imageLabel = ref(null)
const imageLabelSource = ref('')

// 标注叠加摘要：把各模态标注结构归一成「分组 + 条目文本」，便于统一展示
const labelSummary = computed(() => {
  const L = imageLabel.value
  if (!L || typeof L !== 'object') return []
  const groups = []
  const one = (x) => {
    if (x == null) return ''
    if (typeof x !== 'object') return String(x)
    if (x.start !== undefined || x.end !== undefined) return `${x.start ?? ''}~${x.end ?? ''}${x.label ? ' ' + x.label : ''}`
    if (x.time !== undefined) return `${x.time}s${x.label ? ' ' + x.label : ''}`
    if (x.entity) return `${x.entity}${x.type ? '（' + x.type + '）' : ''}`
    if (x.word) return `${x.word}${x.score !== undefined ? ' ' + x.score : ''}`
    if (x.center) return `${JSON.stringify(x.center)}${x.label ? ' ' + x.label : ''}`
    if (x.xyxy) return `${JSON.stringify(x.xyxy)}${x.label ? ' ' + x.label : ''}`
    if (x.frame !== undefined) return `帧${x.frame}${x.time !== undefined ? ' @' + x.time + 's' : ''}${x.label ? ' ' + x.label : ''}`
    return JSON.stringify(x).slice(0, 90)
  }
  const push = (name, arr) => { if (Array.isArray(arr) && arr.length) groups.push({ name, items: arr.map(one) }) }
  push('时间段', L.segments)
  push('检测目标', L.detections)
  push('实体', L.entities)
  push('关键词', L.keywords)
  push('标注项', L.items)
  push('框标注', L.boxes)
  push('通用标注', L.annotations)
  push('标签', L.labels)
  if (!groups.length && L.error) groups.push({ name: '标注错误', items: [String(L.error)] })
  return groups
})

const vizIdx = computed(() =>
  Math.max((viz.value?.files || []).findIndex((f) => f.id === viz.value?.file_id), 0))

async function loadViz(fileId) {
  loading.value = true
  try {
    viz.value = await api.get(`/${props.type}/${props.id}/viz`,
      { params: fileId ? { file_id: fileId } : {} })
    if (showLabel.value) await loadImageLabel()
  } catch { /* 已提示 */ } finally { loading.value = false }
}
function switchFile(idx) {
  const f = viz.value?.files?.[idx]
  if (f) loadViz(f.id)
}
async function loadImageLabel() {
  imagePreview.value = null
  imageLabel.value = null
  if (!showLabel.value || !viz.value?.file_id) return
  try {
    const data = await api.get(`/${props.type}/${props.id}/preview`,
      { params: { file_id: viz.value.file_id } })
    imagePreview.value = data.data
    imageLabel.value = data.label
    imageLabelSource.value = data.label_source || ''
  } catch { /* 无预览数据时静默 */ }
}

const activeTab = ref('overview')
const labelStats = ref({ labels: [], coverage: [], sources: [] })
const statsLoading = ref(false)

const datasetFileCount = computed(() =>
  labelStats.value.total_files ?? viz.value?.files?.length ?? 0)
const labeledFileCount = computed(() =>
  (labelStats.value.sources || [])
    .filter((x) => x.source !== '无标注')
    .reduce((sum, x) => sum + (x.count || 0), 0))

async function loadLabelStats() {
  statsLoading.value = true
  try {
    labelStats.value = await api.get(`/${props.type}/${props.id}/label-stats`)
  } catch { /* 密级不足或无数据时静默 */ } finally { statsLoading.value = false }
}

const labelDistOption = computed(() => {
  const top = (labelStats.value.labels || []).slice(0, 30)
  return {
    grid, tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: top.map(x => x.label), axisLabel: { rotate: 45, color: '#94a3b8' } },
    yAxis: { type: 'value', ...axisStyle },
    series: [{ type: 'bar', data: top.map(x => x.count), itemStyle: { color: '#60a5fa' } }],
  }
})
const sourceOption = computed(() => ({
  tooltip: { trigger: 'item' },
  series: [{ type: 'pie', radius: ['40%', '70%'],
             data: (labelStats.value.sources || []).map(x => ({ name: x.source, value: x.count })),
             label: { color: '#cbd5e1' } }],
}))

onMounted(() => { loadViz(); loadLabelStats() })
</script>

<style scoped>
.preview-bar { display: flex; align-items: center; gap: 10px; }
.viz-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.overview-summary { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin-bottom: 16px; }
.summary-item {
  background: linear-gradient(135deg, var(--dp-bg-card), #16223c);
  border: 1px solid var(--dp-border); border-radius: 10px; padding: 14px 16px;
}
.summary-item span { display: block; color: var(--dp-text-secondary); font-size: 0.78rem; margin-bottom: 6px; }
.summary-item strong { color: var(--dp-primary); font-size: 1.45rem; font-variant-numeric: tabular-nums; }
.overview-wide { grid-column: 1 / -1; }
.dataset-label-list {
  display: flex; flex-wrap: wrap; align-content: flex-start; gap: 8px;
  max-height: 260px; overflow-y: auto; padding-right: 4px;
}
.dataset-label-list .el-tag { margin: 0; }
.label-count { margin-left: 6px; opacity: 0.72; }
h4 { margin: 0 0 10px; font-size: 0.92rem; color: #cbd5e1; }
.frame-strip { display: flex; gap: 10px; overflow-x: auto; padding-bottom: 6px; }
.frame-item { flex-shrink: 0; text-align: center; }
.frame-item img { width: 160px; height: 90px; border-radius: 6px; border: 1px solid #24304d; display: block; }
.frame-item span { font-size: 0.72rem; color: #94a3b8; }
</style>
