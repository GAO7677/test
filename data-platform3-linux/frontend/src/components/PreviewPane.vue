<template>
  <!-- 预览面板：语音=波形+标注段；雷达=BEV点云+标注框；图像=缩略图+标注叠加；视频=原片播放；文本=内容+标注高亮 -->
  <div>
    <div v-if="label?.category || label?.caption" class="meta" style="white-space: pre-wrap; overflow-wrap: anywhere">
      <div v-if="label?.category">类别：{{ label.category }}</div>
      <div v-if="label?.caption">训练文本提示词（caption）：{{ label.caption }}</div>
    </div>
    <template v-if="type === 'voice'">
      <div class="meta">时长 {{ data.duration }}s · 采样率 {{ data.sample_rate }}Hz · 波形 {{ data.waveform?.length }} 点</div>
      <EChart :option="waveOption" height="260px" />
      <div v-if="segments.length" class="label-list">
        <el-tag v-for="(s, i) in segments" :key="i" size="small" effect="plain" type="warning">
          {{ s.start }}s - {{ s.end }}s · {{ s.label || s.text || '片段' }}
        </el-tag>
      </div>
    </template>
    <template v-else-if="type === 'radar'">
      <div class="meta">点云 {{ data.point_count }} 点（已降采样展示 {{ data.points?.length }} 点）</div>
      <EChart :option="bevOption" height="360px" />
      <div v-if="boxes.length" class="label-list">
        <el-tag v-for="(b, i) in boxes" :key="i" size="small" effect="plain" type="warning">
          {{ b.label || '目标' }} <template v-if="b.range">range {{ b.range }}</template>
        </el-tag>
      </div>
    </template>
    <template v-else-if="type === 'image'">
      <div class="meta">分辨率 {{ data.width }}x{{ data.height }}<span v-if="boxes.length"> · {{ boxes.length }} 个标注</span></div>
      <div class="img-wrap">
        <img :src="data.thumbnail" ref="imgRef" @load="drawOverlay" />
        <canvas ref="canvasRef" class="overlay" v-if="boxes.length" />
      </div>
      <div v-if="boxes.length" class="label-list">
        <el-tag v-for="(b, i) in boxes" :key="i" size="small" effect="plain" type="warning">{{ b.label || b.category || '目标' }}</el-tag>
      </div>
    </template>
    <template v-else-if="type === 'video'">
      <div class="meta">
        <div v-if="label?.sampling">抽帧标注：每 {{ label.frame_interval_seconds }} 秒，已处理 {{ label.frames_detected }} 帧；未采样帧未标注。</div>
        <div v-if="label?.truncated" style="color: #e6a23c">达到处理上限，结果仅覆盖至 {{ label.processed_end_time }} 秒，请勿视为全片标注。</div>
        时长 {{ data.duration_seconds ?? '-' }}s · 帧率 {{ data.fps ?? '-' }}fps ·
        分辨率 {{ data.width ?? '?' }}x{{ data.height ?? '?' }}
      </div>
      <VideoPreview v-if="videoSrc" :src="videoSrc" :poster="data.thumbnail" />
      <el-empty v-else description="暂无可播放的视频" :image-size="60" />
      <div v-if="label" class="meta">此处播放原视频；逐帧框请进入在线标注查看与修改。</div>
    </template>
    <template v-else-if="type === 'text'">
      <div class="meta">共 {{ data.line_count }} 行 · 编码 {{ data.encoding }} · 最多显示20万字符</div>
      <pre class="text-preview" v-if="!spans.length">{{ data.text ?? (data.lines || []).join('\n') }}</pre>
      <pre v-else class="text-preview"><template v-for="(seg, i) in textSegments" :key="i"><mark v-if="seg.mark" class="hl" :title="seg.label">{{ seg.text }}</mark><template v-else>{{ seg.text }}</template></template></pre>
      <div v-if="spans.length" class="label-list">
        <el-tag v-for="(s, i) in spans" :key="i" size="small" effect="plain" type="warning">{{ s.label }}：{{ s.text || `[${s.start},${s.end}]` }}</el-tag>
      </div>
    </template>
  </div>
</template>

<script setup>
import { computed, nextTick, ref, watch } from 'vue'
import EChart from './EChart.vue'
import VideoPreview from './VideoPreview.vue'

const props = defineProps({ type: String, data: Object, videoSrc: { type: String, default: '' }, label: { type: [Object, Array], default: null } })

// ---- 标注归一化：自动标注与手动标注产出均为数组或含数组字段 ----
const items = computed(() => {
  const l = props.label
  if (!l) return []
  if (Array.isArray(l)) return l
  for (const k of ['items', 'annotations', 'boxes', 'segments', 'spans', 'entities', 'detections', 'labels']) {
    if (Array.isArray(l[k])) return l[k]
  }
  return []
})
// 框坐标归一化：兼容三种产物格式——xyxy/bbox 数组、type 标记，以及
// coco_detect/边缘检测等算法输出的顶层 x,y,w,h（一律转成 xyxy 供绘制）
function normBox(b) {
  if (b.xyxy || b.bbox) return b
  if (typeof b.x === 'number' && typeof b.y === 'number' &&
      typeof b.w === 'number' && typeof b.h === 'number') {
    return { ...b, xyxy: [b.x, b.y, b.x + b.w, b.y + b.h] }
  }
  return b
}
const boxes = computed(() => items.value
  .filter((x) => x && (x.xyxy || x.bbox || x.type === 'bbox' || x.type === 'bev_box' ||
    (typeof x.x === 'number' && typeof x.w === 'number' && props.type === 'image')))
  .map(normBox))
const segments = computed(() => items.value.filter((x) => x && (x.type === 'segment' || (x.start != null && x.end != null && props.type === 'voice'))))
const spans = computed(() => items.value.filter((x) => x && (x.type === 'span' || (x.start != null && x.end != null && props.type === 'text'))))

// ---- 图像标注叠加（Canvas 画框） ----
const imgRef = ref(null)
const canvasRef = ref(null)
function drawOverlay() {
  const img = imgRef.value
  const cv = canvasRef.value
  if (!img || !cv) return
  cv.width = img.clientWidth
  cv.height = img.clientHeight
  const ctx = cv.getContext('2d')
  ctx.clearRect(0, 0, cv.width, cv.height)
  const sx = cv.width / (props.data.width || img.naturalWidth || 1)
  const sy = cv.height / (props.data.height || img.naturalHeight || 1)
  for (const b of boxes.value) {
    const xy = b.xyxy || b.bbox
    if (!xy || xy.length < 4) continue
    const [x1, y1, x2, y2] = xy
    ctx.strokeStyle = '#fbbf24'
    ctx.lineWidth = 2
    ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy)
    const text = b.label || b.category || ''
    if (text) {
      ctx.fillStyle = 'rgba(251,191,36,0.85)'
      ctx.font = '12px sans-serif'
      const w = ctx.measureText(text).width + 8
      ctx.fillRect(x1 * sx, Math.max(y1 * sy - 18, 0), w, 16)
      ctx.fillStyle = '#0c1220'
      ctx.fillText(text, x1 * sx + 4, Math.max(y1 * sy - 5, 12))
    }
  }
}
watch(() => [props.label, props.data], () => nextTick(drawOverlay), { deep: true })

// ---- 文本标注高亮分段 ----
const textSegments = computed(() => {
  const text = props.data.text ?? (props.data.lines || []).join('\n')
  if (!spans.value.length) return []
  const marks = spans.value.map((a) => a.offset_unit === 'utf16' ? a : {
      ...a, start: [...text].slice(0,a.start).join('').length,
      end: [...text].slice(0,a.end).join('').length, label: a.label || a.type,
    })
    .filter((s) => Number.isInteger(s.start) && Number.isInteger(s.end) && s.end > s.start)
    .sort((a, b) => a.start - b.start)
  const segs = []
  let pos = 0
  for (const m of marks) {
    if (m.start < pos || m.start > text.length) continue
    if (m.start > pos) segs.push({ text: text.slice(pos, m.start) })
    segs.push({ text: text.slice(m.start, Math.min(m.end, text.length)), mark: true, label: m.label })
    pos = Math.min(m.end, text.length)
  }
  if (pos < text.length) segs.push({ text: text.slice(pos) })
  return segs
})

// ---- 语音波形 + 标注段标记 ----
const waveOption = computed(() => {
  const series = [{
    type: 'line', showSymbol: false, lineStyle: { color: '#60a5fa', width: 1 },
    areaStyle: { color: 'rgba(96,165,250,0.2)' },
    data: (props.data.waveform || []).map((p) => p[1]),
  }]
  const duration = props.data.duration || 0
  const n = (props.data.waveform || []).length
  if (segments.value.length && duration && n) {
    series[0].markArea = {
      itemStyle: { color: 'rgba(251,191,36,0.18)' },
      data: segments.value.map((s) => [
        { xAxis: Math.round((s.start / duration) * (n - 1)), name: s.label || '' },
        { xAxis: Math.round((s.end / duration) * (n - 1)) },
      ]),
    }
  }
  return {
    grid: { left: 50, right: 20, top: 20, bottom: 30 },
    xAxis: { type: 'category', show: false, data: (props.data.waveform || []).map((_, i) => i) },
    yAxis: { type: 'value', splitLine: { lineStyle: { color: '#24304d' } } },
    series,
  }
})

const bevOption = computed(() => {
  const series = [{
    type: 'scatter', symbolSize: 3,
    itemStyle: { color: '#60a5fa', opacity: 0.7 },
    data: (props.data.points || []).map((p) => [p[0], p[1]]),
  }]
  // BEV 标注框（range/azimuth 或 xyxy）
  for (const b of boxes.value) {
    let x1, x2, y1, y2
    if (b.range && b.azimuth) { [x1, x2] = b.range; [y1, y2] = b.azimuth }
    else if (b.xyxy) { [x1, y1, x2, y2] = b.xyxy }
    else continue
    series.push({
      type: 'line', data: [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]],
      showSymbol: false, lineStyle: { color: '#fbbf24', width: 2 }, silent: true,
    })
  }
  return {
    grid: { left: 50, right: 20, top: 20, bottom: 40 },
    xAxis: { type: 'value', name: 'X (m)', splitLine: { lineStyle: { color: '#24304d' } } },
    yAxis: { type: 'value', name: 'Y (m)', splitLine: { lineStyle: { color: '#24304d' } } },
    series,
  }
})
</script>

<style scoped>
.meta { color: #94a3b8; font-size: 0.82rem; margin-bottom: 10px; }
.text-preview {
  background: #0f1729; border: 1px solid #24304d; border-radius: 8px;
  padding: 12px; max-height: 400px; overflow: auto;
  font-size: 0.78rem; color: #cbd5e1; white-space: pre-wrap;
}
.img-wrap { position: relative; text-align: center; }
.img-wrap img { max-width: 100%; border-radius: 8px; display: block; margin: 0 auto; }
.overlay { position: absolute; left: 50%; transform: translateX(-50%); top: 0; pointer-events: none; }
.label-list { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 6px; }
.hl { background: rgba(251, 191, 36, 0.35); color: #fbbf24; border-radius: 2px; padding: 0 1px; }
</style>
