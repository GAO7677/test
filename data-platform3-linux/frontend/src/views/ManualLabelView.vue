<template>
  <div class="page">
    <div class="page-header">
      <h2><span>✍️</span> 标注工作台</h2>
      <p>选择文件进行标注或修正已有结果；保存为新版本，保留原始数据。</p>
    </div>

    <div class="filter-bar">
      <el-select v-model="datasetId" placeholder="选择数据集" filterable style="width: 260px" @change="onDatasetChange">
        <el-option v-for="d in datasets" :key="d.id" :label="`${d.name}（${TYPE_NAMES[d.data_type]}）`" :value="d.id" />
      </el-select>
      <el-select v-model="inputVersionId" placeholder="输入版本" filterable style="width: 200px" @change="loadFiles">
        <el-option v-for="v in versions" :key="v.id" :value="v.id" :disabled="v.ready === false"
                   :label="`${v.version}（${v.stage} · ${v.ready === false ? '未完成或失败' : (v.status === 'published' ? '已发布' : '草稿')}）`" />
      </el-select>
      <el-tag effect="plain">{{ labelType }}</el-tag>
      <div v-if="modality !== 'video'" style="margin-left: auto">
        <el-button type="primary" :loading="submitting"
                   @click="onSubmit">
          💾 提交标注（{{ changedCount }} 条）
        </el-button>
      </div>
    </div>

    <div class="table-card" v-loading="loading">
      <div style="display: flex; gap: 10px; margin-bottom: 10px">
        <el-input v-model="fileSearch" placeholder="按文件名搜索…" clearable style="width: 240px"
                  @change="loadFiles(1)" @clear="loadFiles(1)" />
        <span style="margin-left: auto; font-size: 0.8rem; color: var(--text-secondary); align-self: center">
          共 {{ fileTotal }} 个文件
        </span>
      </div>
      <el-table :data="files" size="default" max-height="520">
        <el-table-column prop="name" label="文件" min-width="220" show-overflow-tooltip>
          <template #default="{ row }">
            <span :title="row.path">{{ row.name }}</span>
          </template>
        </el-table-column>
        <el-table-column label="大小" width="110"><template #default="{ row }">{{ fmtSize(row.size) }}</template></el-table-column>
        <el-table-column label="标注数" width="90">
          <template #default="{ row }">
            <el-tag size="small" :type="labelCount(row) ? 'success' : 'info'" effect="plain">
              {{ labelCount(row) ? labelCount(row) + ' 条' : '未标注' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="120">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="openWorkbench(row)">🎯 开始标注</el-button>
          </template>
        </el-table-column>
        <template #empty><div class="empty-tip">请选择数据集与版本查看文件</div></template>
      </el-table>
      <div class="pager" style="margin-top: 10px; display: flex; justify-content: flex-end">
        <el-pagination background layout="total, sizes, prev, pager, next" :total="fileTotal"
                       v-model:current-page="filePage" v-model:page-size="filePageSize"
                       :page-sizes="[20, 50, 100, 200]" @change="loadFiles()" />
      </div>
    </div>

    <!-- ============ 标注工作台 ============ -->
    <el-dialog v-model="wbVisible" :title="`标注工作台 — ${currentFile?.name || ''}`"
               width="92%" top="3vh" destroy-on-close class="wb-dialog" :before-close="closeWorkbench" @closed="stopVideoCallback">
      <div v-if="wbLoading" v-loading="true" style="height: 300px" />
      <template v-else>
        <div v-if="modality !== 'video'" class="wb-toolbar">
          <span style="font-size: 0.82rem; color: var(--text-secondary)">{{ toolHint }}</span>
          <el-input v-model="currentLabel" placeholder="类别标签（画框/划选后赋予）" size="small" style="width: 200px" />
          <el-button size="small" type="danger" plain :disabled="!selectedAnn" @click="removeSelected">删除选中</el-button>
          <el-button v-if="autoCount > 0" size="small" type="warning" plain
                     @click="clearAuto">清除自动标注（{{ autoCount }}）</el-button>
          <el-button size="small" plain @click="clearAll">清空全部</el-button>
          <div style="margin-left: auto">
            <el-button size="small" @click="closeWorkbench()">关闭</el-button>
            <el-button size="small" type="primary" @click="saveFile">保存本文件标注（{{ anns.length }}）</el-button>
          </div>
        </div>

        <!-- 图像：拖框 -->
        <div v-if="modality === 'image'" class="stage" ref="stageRef">
          <img :src="fileUrl" @load="onImgLoad" ref="imgEl" draggable="false" />
          <canvas ref="cvEl" class="draw-layer"
                  @mousedown="onBoxStart" @mousemove="onBoxMove" @mouseup="onBoxEnd" @mouseleave="onBoxEnd" />
        </div>

        <!-- 视频：暂停后拖框，标注挂在播放时间点 -->
        <div v-else-if="modality === 'video'">
          <div class="wb-toolbar">
            <strong>1. 标注方式</strong>
            <el-radio-group v-model="videoWorkflow" :disabled="videoLocked || videoPreview" @change="changeWorkflow">
              <el-radio-button value="manual">逐帧手动标注</el-radio-button>
              <el-radio-button value="first_frame_auto">首帧手动标注＋后续自动标注</el-radio-button>
            </el-radio-group>
          </div>
          <div v-if="!videoPreview" class="wb-toolbar">
            <strong>2. 生成的标注文件（多选）</strong>
            <el-checkbox-group v-model="outputTypes" :disabled="videoLocked || videoPreview" @change="outputsChanged">
              <el-checkbox value="box">目标框</el-checkbox><el-checkbox value="mask">分割 mask</el-checkbox><el-checkbox value="trajectory">点轨迹</el-checkbox>
            </el-checkbox-group>
          </div>
          <div v-else class="wb-toolbar preview-layers">
            <strong>预览叠加（多选）</strong>
            <el-checkbox-group v-model="previewTypes" @change="redraw">
              <el-checkbox v-for="type in outputTypes" :key="type" :value="type">{{ previewTypeNames[type] }}</el-checkbox>
            </el-checkbox-group>
            <span class="hint">仅切换显示，不改变生成文件；取消全部勾选可看原视频。</span>
          </div>
          <div v-if="!videoPreview" class="wb-toolbar">
            <strong>3. 类别与画框</strong>
            <el-select :model-value="currentLabel" @change="chooseCategory" :disabled="videoLocked" filterable size="small" style="width:230px" placeholder="选择类别，或在顶部自定义">
              <el-option :value="null" label="＋ 自定义类别" />
              <el-option v-for="c in videoClasses" :key="c" :label="c" :value="c" />
            </el-select>
            <el-radio-group v-model="videoEditMode" size="small" :disabled="videoLocked" @change="setVideoTool">
              <el-radio-button value="box">{{ videoDraft?.mask ? '修改结果框' : '画提示框' }}</el-radio-button>
              <el-radio-button value="brush" :disabled="!videoDraft?.mask || seedPending">添加 mask</el-radio-button>
              <el-radio-button value="erase" :disabled="!videoDraft?.mask || seedPending">擦除 mask</el-radio-button>
              <el-radio-button v-if="outputTypes.includes('trajectory')" value="point" :disabled="!videoDraft?.mask || seedPending">调整采样点</el-radio-button>
            </el-radio-group>
            <span class="hint">{{ videoWorkflow==='manual' ? '逐帧画框并确认；右侧记录已确认的人工标注。' : '先确认所有首帧物体，再统一启动后台自动标注。' }}</span>
          </div>
          <div v-if="!videoPreview && ['brush','erase'].includes(videoEditMode)" class="wb-toolbar">
            <span>画笔直径（原图像素）</span><el-input-number v-model="maskBrushSize" :min="1" :max="160" size="small" :disabled="videoLocked" />
            <span class="hint">修改 mask 后更新默认外接框；手动修改过的结果框保留。</span>
          </div>
          <div class="video-workspace">
          <div class="video-picture">
          <div class="stage" ref="stageRef" style="display: inline-block">
            <video :src="fileUrl" ref="videoEl" style="display:block; max-width:100%; max-height:52vh"
                   @play="animateVideo" @timeupdate="redraw" @pause="syncVideoCanvas" @seeked="syncVideoCanvas" @loadedmetadata="syncVideoCanvas" />
            <canvas ref="cvEl" class="draw-layer" style="cursor:crosshair"
                    @mousedown="onBoxStart" @mousemove="onBoxMove" @mouseup="onBoxEnd" @mouseleave="onBoxEnd" />
          </div>
          <div class="wb-toolbar video-timeline" style="margin-top:8px">
            <el-button size="small" :disabled="videoTimelineLocked" @click="toggleVideo">播放 / 暂停</el-button>
            <el-button size="small" :disabled="videoTimelineLocked || videoFrame===0" @click="seekVideoFrame(0)">首帧</el-button>
            <el-button size="small" :disabled="videoTimelineLocked || videoFrame===0" @click="stepFrame(-1)">上一帧</el-button>
            <el-button size="small" :disabled="videoTimelineLocked || videoFrame>=videoMeta.frame_count-1" @click="stepFrame(1)">下一帧</el-button>
            <span>帧 {{ videoFrame }} / {{ Math.max(0,(videoMeta.frame_count || 1)-1) }}</span>
            <el-slider :model-value="videoFrame" :disabled="videoTimelineLocked || videoMeta.frame_count<=1" :min="0" :max="Math.max(1,(videoMeta.frame_count || 1)-1)" @input="seekVideoFrame" style="flex:1;min-width:160px;margin:0 12px" />
            <el-checkbox v-if="!videoPreview && outputTypes.includes('mask')" v-model="showMask" @change="redraw">显示 mask</el-checkbox>
          </div>
          <div class="hint" v-if="videoPreview">{{ previewStatus }}</div>
          </div>
          <aside class="video-results">
            <section v-if="!videoPreview" class="current-annotation">
              <h4>当前标注</h4>
              <el-alert class="annotation-status" :title="annotationStatus.title" :type="annotationStatus.type" :closable="false" show-icon role="status" aria-live="polite" />
              <template v-if="videoDraft">
                <dl><dt>类别</dt><dd>{{ videoDraft.label }}</dd><dt>帧</dt><dd>{{ videoDraft.frame }}</dd><dt>框位置</dt><dd>{{ boxText(videoDraft) }}</dd></dl>
                <p class="hint">原图像素坐标：(x1, y1, x2, y2)。SAM 就绪后重画只修改结果框，保留 mask 和点。</p>
                <p v-if="videoDraft.seed_status==='failed'" class="draft-error">{{ videoDraft.seed_error }} <el-button link type="primary" @click="queueSeed(videoDraft)">重试 SAM</el-button></p>
                <template v-if="outputTypes.includes('trajectory')">
                  <div class="point-settings"><span>点数上限</span><el-input-number :model-value="videoDraft.point_limit || 8" :min="1" :max="32" size="small" :disabled="videoLocked" @change="setPointLimit" /><span>{{ videoDraft.tracking_points?.length || 0 }} 个点</span></div>
                  <p class="hint">在上方选择「调整采样点」：点击已有点删除，点击画面添加。确认修改后再启动自动标注。</p>
                </template>
                <div class="draft-actions">
                  <el-button type="primary" size="small" :disabled="videoLocked || seedPending || !draftReady || !outputTypes.length" @click="confirmVideoDraft">{{ editingVideoKey ? '确认修改' : '确认标注' }}</el-button>
                  <el-button size="small" :disabled="videoLocked" @click="cancelVideoDraft">取消</el-button>
                </div>
              </template>
            </section>
            <section class="confirmed-annotations">
              <h4>{{ videoPreview ? '结果预览' : '已标注结果（人工）' }} <span class="hint">{{ videoPreview ? `第 ${videoFrame} 帧` : `${humanResults.length} 条` }}</span></h4>
              <p class="hint">{{ videoPreview ? '点击结果可进入人工修正；自动生成的内容仍需核验。' : (videoWorkflow==='manual' ? '按帧记录；可预览、修改或删除单条标注。' : '这里仅列出已确认的首帧标注。') }}</p>
              <div class="annotation-result-list">
                <template v-if="videoPreview">
                  <button v-for="{a,i} in visibleAnnotations.filter(({a})=>a.xyxy)" :key="i" class="annotation-result" @click="editVideoAnnotation(a)"><strong>{{ a.label }}</strong><span>{{ boxText(a) }}</span><small>{{ a.source==='manual' ? '人工标注' : '自动生成 · 待核验' }}</small></button>
                </template>
                <template v-else>
                  <div v-for="a in humanResults" :key="annotationKey(a)" class="annotation-result" :class="{editing:editingVideoKey===annotationKey(a),previewing:previewedVideoAnnotation && annotationKey(previewedVideoAnnotation)===annotationKey(a)}">
                    <strong>{{ a.label }} <small>第 {{ a.frame }} 帧</small></strong><span>{{ boxText(a) }}</span>
                    <small>{{ editingVideoKey===annotationKey(a) ? '正在修改，原结果保留' : (previewedVideoAnnotation && annotationKey(previewedVideoAnnotation)===annotationKey(a) ? '预览中' : '已确认') }}</small>
                    <el-tag v-if="videoWorkflow==='first_frame_auto'" size="small" :type="objectStatus(a).type" disable-transitions>{{ objectStatus(a).title }}</el-tag>
                    <small v-if="a.generation?.error" class="draft-error">{{ a.generation.error }}</small>
                    <div class="annotation-actions">
                      <el-button size="small" :disabled="videoLocked || !!videoDraft || (videoWorkflow==='first_frame_auto' && !objectResultReady(a))" @click="previewVideoAnnotation(a)">预览</el-button>
                      <el-button size="small" type="primary" plain :disabled="videoLocked || !!videoDraft" @click="editVideoAnnotation(a)">修改</el-button>
                      <el-button size="small" type="danger" plain :disabled="videoLocked || !!videoDraft" @click="deleteVideoAnnotation(a)">删除</el-button>
                    </div>
                  </div>
                  <p v-if="!humanResults.length" class="hint">尚无已确认的人工标注。</p>
                </template>
              </div>
            </section>
          </aside>
          </div>
          <VideoAssist @status="updateObjectJob" @completed="completeObjectGeneration" :key="`${datasetId}:${inputVersionId}:${keyOf(currentFile || {})}`" ref="videoAssist" :disabled="submitting" @busy="videoBusy=$event" @submitted="videoSubmitted" />
          <div v-if="!videoPreview && videoWorkflow==='first_frame_auto'" class="wb-toolbar">
            <span>自动处理首帧至第</span><el-input-number v-model="autoEndFrame" :min="0" :max="Math.min(299,(videoMeta.frame_count || 1)-1)" :disabled="videoLocked" size="small" /><span>帧（点击后台启动时生效，单次最多 300 帧）</span>
          </div>
          <div class="wb-toolbar" style="justify-content:flex-end;margin-top:12px">
            <el-tag v-if="videoPreview" type="success" disable-transitions>结果预览 · 待人工核验</el-tag>
            <el-button v-if="videoPreview" size="small" @click="togglePreview">{{ objectPreviewId!==null ? '返回首帧标注' : '继续修正' }}</el-button>
            <el-button v-if="videoPreview" size="small" @click="downloadVideoLabels">下载标注文件</el-button>
            <el-button size="small" :disabled="submitting || backgroundSubmitting || objectLoading" @click="closeWorkbench()">{{ videoBusy ? '关闭，后台继续' : '关闭' }}</el-button>
            <el-button v-if="!videoPreview && videoWorkflow==='first_frame_auto'" size="small" type="primary" :disabled="videoLocked || !!videoDraft || !humanResults.length || !outputTypes.length" @click="generateConfirmed">{{ hasAutomaticResult ? '重新自动标注' : '开始后台自动标注' }}</el-button>
            <el-button v-if="!videoPreview" size="small" type="primary" :disabled="videoLocked || !!videoDraft || !outputTypes.length" :loading="submitting" @click="finishVideo">
              {{ videoWorkflow==='manual' ? '保存并预览' : '保存当前标注' }}
            </el-button>
          </div>
        </div>

        <!-- 语音：波形段选 -->
        <div v-else-if="modality === 'voice'">
          <audio :src="fileUrl" controls style="width: 100%" />
          <canvas ref="waveEl" class="wave-canvas"
                  @mousedown="onSegStart" @mousemove="onSegMove" @mouseup="onSegEnd" />
          <div style="font-size: 0.78rem; color: var(--text-secondary); margin-top: 6px">
            在波形上按住拖动选择时间段；点击已标注段选中
          </div>
        </div>

        <!-- 文本：划词 -->
        <div v-else-if="modality === 'text'">
          <div class="text-stage" ref="textEl" @mouseup="onTextSelect"><template v-for="(seg, i) in textRender" :key="i"><mark v-if="seg.mark" class="hl" :class="{ sel: seg.idx === selectedIdx }" @click="selectedIdx = seg.idx">{{ seg.text }}</mark><template v-else>{{ seg.text }}</template></template></div>
          <div style="font-size: 0.78rem; color: var(--text-secondary); margin-top: 6px">
            用鼠标划选文字即生成实体标注；点击高亮片段选中
          </div>
        </div>

        <!-- 雷达：BEV 框选 -->
        <div v-else-if="modality === 'radar'">
          <canvas ref="bevEl" class="bev-canvas"
                  @mousedown="onBoxStart" @mousemove="onBoxMove" @mouseup="onBoxEnd" @mouseleave="onBoxEnd" />
          <div style="font-size: 0.78rem; color: var(--text-secondary); margin-top: 6px">
            在点云俯视图上拖框圈选目标（记录距离/方位范围）
          </div>
        </div>

        <div v-if="selectedAnn && modality !== 'video'" style="display: flex; gap: 12px; margin: 12px 0">
          <el-input v-model="selectedAnn.label" placeholder="类别" style="width: 220px" @input="redraw" />
          <el-input v-if="selectedAnn.type === 'segment'" v-model="selectedAnn.text" placeholder="转写文本" />
        </div>
        <!-- 已标注条目 -->
        <div v-if="anns.length && modality!=='video'" class="ann-list">
          <el-tag v-for="{a, i} in visibleAnnotations" :key="i" size="small" effect="plain" disable-transitions
                  :type="i === selectedIdx ? 'warning' : 'info'" @click="selectAnnotation(i)" style="cursor: pointer">
            {{ annDesc(a) }}
          </el-tag>
        </div>
      </template>
    </el-dialog>

    <el-collapse style="margin-top:16px"><el-collapse-item title="标注历史与任务" name="history">
    <div v-if="modality === 'video'" class="table-card">
      <h4>视频模型标注任务</h4>
      <TaskTable ref="videoTasks" url="/video-annotation" :version-map="versionMap">
        <el-table-column prop="name" label="任务" min-width="200" />
        <el-table-column label="操作" width="180"><template #default="{row}">
          <el-button v-if="row.status === 'completed'" link type="primary" @click="openVideoResult(row)">预览结果</el-button>
          <el-button v-if="row.workflow && ['failed','cancelled'].includes(row.status)" link type="primary" @click="openVideoInputs(row)">修改并重试</el-button>
          <el-button v-if="['pending','running'].includes(row.status)" link type="danger" @click="cancelVideoTask(row)">取消</el-button>
        </template></el-table-column>
      </TaskTable>
    </div>
    <div class="table-card" style="margin-top: 16px">
      <h4 style="margin: 0 0 10px">历史手动标注任务</h4>
      <TaskTable ref="manualTasks" url="/manual-label" :has-progress="false" :version-map="versionMap">
        <el-table-column prop="label_type" label="标注类型" width="120" />
      </TaskTable>
    </div>
    </el-collapse-item></el-collapse>
  </div>
</template>

<script setup>
import { computed, nextTick, onMounted, onBeforeUnmount, ref, watch } from 'vue'
import { onBeforeRouteLeave } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { clone, decodeLabel, encodeLabel, encodeVideoLabel, videoAnnotationKey } from '../annotationAdapter'
import api from '../api'
import { useDatasets } from '../composables'
import { fmtSize, TYPE_NAMES } from '../utils'
import TaskTable from '../components/TaskTable.vue'
import VideoAssist from '../components/VideoAssist.vue'
import {maskCanvas,canvasMask,sampleMaskPoints,hasMaskPixels,paintMaskStroke,pointsInMask} from '../videoMask'

const { datasets, versionMap, datasetVersions, loadDatasets, currentVersionId, routeInput } = useDatasets()
const datasetId = ref(null)
const inputVersionId = ref(null)
const labelType = ref('目标检测')
const files = ref([])
const loading = ref(false)
const submitting = ref(false)
const labelsMap = ref({})        // file -> 标注数组（含已有与新标）
const adapters = ref({})
let loadedContext = null
let fileRequest = 0
const changed = ref(new Set())   // 本次改动的文件

const versions = computed(() =>
  [...(datasetVersions.value[datasetId.value] || [])].sort((a, b) => a.version_no - b.version_no))
const currentDataset = computed(() => datasets.value.find((d) => d.id === datasetId.value))
const modality = computed(() => currentDataset.value?.data_type || 'image')
const changedCount = computed(() => changed.value.size)
const toolHint = computed(() => ({
  image: '在图像上按住拖动绘制检测框；点击框可选中',
  video: '暂停视频 → 拖框 → 记录当前时间点的目标框',
  voice: '在波形上拖动选择时间段进行标注',
  text: '划选文字生成实体标注',
  radar: '在 BEV 点云图上拖框圈选目标',
}[modality.value] || ''))

// 文件唯一键 = 版本目录内相对路径（文件夹上传场景下同名文件靠目录区分）
function keyOf(row) { return row.path || row.name }
function fileUrlOf(key) {
  return `/api/datasets/${datasetId.value}/versions/${inputVersionId.value}/file?path=${encodeURIComponent(key)}`
}
function labelKeyOf(key) {
  const dot = key.lastIndexOf('.')
  return (dot > 0 ? key.slice(0, dot) : key) + '_label.json'
}

const fileSearch = ref('')
const filePage = ref(1)
const filePageSize = ref(50)
const fileTotal = ref(0)

async function onDatasetChange() {
  labelType.value = ({ text: '实体识别', voice: '语音识别', video: '视频标注' })[modality.value] || '目标检测'
  inputVersionId.value = currentVersionId(datasetId.value)
  await loadFiles(1)
}

async function loadFiles(resetPage) {
  if (!datasetId.value || !inputVersionId.value) return
  const context = [datasetId.value, inputVersionId.value]
  if (loadedContext && context.join(':') !== loadedContext.join(':')) {
    if (changed.value.size) {
      try { await ElMessageBox.confirm('切换将放弃尚未提交的标注，是否继续？', '未提交标注') }
      catch { [datasetId.value, inputVersionId.value] = loadedContext; return }
    }
    labelsMap.value = {}; adapters.value = {}; changed.value = new Set()
    wbVisible.value = false; currentFile.value = null; anns.value = []
  }
  loadedContext = context
  const requestId = ++fileRequest
  if (resetPage) filePage.value = 1
  loading.value = true
  try {
    const data = await api.get(`/datasets/${datasetId.value}/versions/${inputVersionId.value}/files`, {
      params: { search: fileSearch.value, page: filePage.value, page_size: filePageSize.value },
    })
    if (requestId !== fileRequest) return
    files.value = data.items
    fileTotal.value = data.total
  } finally { loading.value = false }
}

function labelCount(row) {
  const k = keyOf(row)
  if (labelsMap.value[k]) return labelsMap.value[k].length
  return row.label_count || 0
}

// ============ 标注工作台 ============
const wbVisible = ref(false)
const wbLoading = ref(false)
const currentFile = ref(null)
const anns = ref([])             // 当前文件标注数组
const currentLabel = ref('')
const selectedIdx = ref(-1)
const selectedAnn = computed(() => modality.value==='video' ? videoDraft.value : anns.value[selectedIdx.value])
const fileUrl = ref('')
const videoMeta = ref({}), videoClasses = ref([]), videoFrame = ref(0), videoTasks = ref(null)
const videoEditMode = ref('box'), showMask = ref(true)
const videoWorkflow=ref('manual'), outputTypes=ref(['box']), autoEndFrame=ref(0), videoAssist=ref(null)
const previewedVideoAnnotation=ref(null),objectPreviewId=ref(null),activeObjectKeys=ref([]),objectLoading=ref(false),backgroundSubmitting=ref(false)
const maskBrushSize=ref(12)
let maskStroke=null
// Derived status; no API calls or mutations.
function objectResultReady(a){
  const g=a.generation
  return g?.status==='completed' && g.end_frame===autoEndFrame.value && g.output_modes?.length===outputTypes.value.length && outputTypes.value.every(type=>g.output_modes.includes(type))
}
function objectStatus(a){
  const g=a.generation||{}
  if(['pending','running'].includes(g.status))return {type:'warning',title:g.status==='pending'?'排队中':'正在自动标注…'}
  if(g.status==='failed')return {type:'danger',title:'生成失败，请重试'}
  if(g.status==='cancelled')return {type:'info',title:'已取消，可重试'}
  if(g.status==='stale')return {type:'warning',title:'首帧已修改，需重新生成'}
  return objectResultReady(a)?{type:'success',title:'生成完成 · 可预览'}:{type:'warning',title:g.status==='completed'?'设置已变化，需重新生成':'已准备，待后台启动'}
}
const annotationStatus=computed(()=>{
  if(objectLoading.value)return {type:'warning',title:'正在加载自动标注结果…'}
  if(videoBusy.value)return {type:'warning',title:'后台正在自动标注，可关闭工作台，在任务列表查看进度'}
  if(seedPending.value)return {type:'warning',title:'SAM 正在生成 mask…'}
  const a=videoDraft.value
  if(a?.seed_status==='failed')return {type:'error',title:'SAM 生成失败，请重试或重画'}
  if(a?.mask && !hasMaskPixels(a.mask))return {type:'error',title:'mask 已擦空，请添加有效区域'}
  if(a?.mask)return {type:'success',title:'mask 已就绪，可修正后确认标注'}
  if(a)return {type:'warning',title:'当前标注待确认'}
  return {type:'info',title:'请选择类别并画框，或修改已有标注'}
})
const videoDraft=ref(null), editingVideoKey=ref(null), manualVideoResults=ref([]), openedManualSnapshot=ref('[]')
const annotationKey=videoAnnotationKey
const boxText=a=>a?.xyxy ? `(${a.xyxy.map(x=>Math.round(x*10)/10).join(', ')})` : '尚未画框'
const humanResults=computed(()=>manualVideoResults.value.filter(a=>videoWorkflow.value!=='first_frame_auto' || a.frame===0).slice().sort((a,b)=>a.frame-b.frame || a.track_id-b.track_id))
const hasAutomaticResult=computed(()=>{
  const original=adapters.value[keyOf(currentFile.value || {})]?.original
  return !!original?.tracking_runs?.length || [...manualVideoResults.value,...(original?.manual_annotations || [])].some(a=>a.generation?.status==='completed')
})
const seedPending=computed(()=>['queued','running'].includes(videoDraft.value?.seed_status))
const draftReady=computed(()=>{
  const a=videoDraft.value
  if(!a?.label?.trim() || !a.xyxy)return false
  if(!hasMaskPixels(a.mask))return false
  return !outputTypes.value.includes('trajectory') || !!a.tracking_points?.length
})
const encodedVideoLabel=computed(()=>encodeVideoLabel(
  adapters.value[keyOf(currentFile.value || {})],anns.value,manualVideoResults.value,
  {export_types:outputTypes.value,workflow:videoWorkflow.value,fps:videoFps(),width:videoMeta.value.width,height:videoMeta.value.height}
))
const visibleAnnotations = computed(() => {
  const rows=anns.value.map((a,i)=>({a,i})).filter(({a})=>objectPreviewId.value===null || (a.track_id===objectPreviewId.value && a.frame<=autoEndFrame.value))
  if(modality.value!=='video')return rows
  const objects=new Map(),points=[]
  for(const row of rows.filter(({a})=>a.frame===videoFrame.value)){
    if(row.a.type==='video_point'){if(videoPreview.value && previewTypes.value.includes('trajectory'))points.push(row);continue}
    const prev=objects.get(row.a.track_id)
    if(!prev || row.a.mask)objects.set(row.a.track_id,row)
  }
  return [...objects.values(),...points]
})
const videoPreview=ref(false),previewTypes=ref([])
const previewTypeNames={box:'目标框',mask:'分割 mask',trajectory:'点轨迹'}
watch([videoPreview,()=>outputTypes.value.join('|')],([active])=>{
  if(active){previewTypes.value=[...outputTypes.value];nextTick(redraw)}
})
let initialPreviewFrame=null
const videoBusy=ref(false), manualTasks=ref(null), openedSnapshot=ref(''), modelSnapshot=ref(null)
const videoLocked=computed(()=>videoBusy.value || submitting.value || objectLoading.value || backgroundSubmitting.value)
const videoTimelineLocked=computed(()=>videoLocked.value || !!videoDraft.value || (!videoPreview.value && videoWorkflow.value==='first_frame_auto'))
const workbenchDirty=computed(()=>JSON.stringify(anns.value)!==openedSnapshot.value || (modality.value==='video' && (!!videoDraft.value || JSON.stringify(manualVideoResults.value)!==openedManualSnapshot.value)))
const maskCache = new Map()
function pauseVideo(){onBoxEnd();videoEl.value?.pause()}
function videoSubmitted(){
  modelSnapshot.value=JSON.stringify(anns.value);openedSnapshot.value=modelSnapshot.value
  openedManualSnapshot.value=JSON.stringify(manualVideoResults.value)
  videoTasks.value?.reload()
  ElMessage.success('已提交后台，可关闭工作台；在“标注历史与任务”查看进度和结果')
}
const previewStatus=computed(()=>{
  if(!previewTypes.value.length)return '已隐藏所有叠加，仅显示原始视频。'
  const rows=visibleAnnotations.value.map(({a})=>a)
  return rows.length ? `当前帧 ${rows.filter(a=>a.type!=='video_point').length} 个物体、${rows.filter(a=>a.type==='video_point').length} 个轨迹点；不可见目标不绘制。` : '当前帧尚无标注。'
})
// User actions: draft -> confirmed annotation -> optional model generation.
function togglePreview(){pauseVideo();videoPreview.value=false;objectPreviewId.value=null;videoEditMode.value='box';if(videoWorkflow.value==='first_frame_auto')seekVideoFrame(0);redraw()}
function changeWorkflow(){
  pauseVideo()
  if(videoDraft.value){videoWorkflow.value=videoWorkflow.value==='manual'?'first_frame_auto':'manual';ElMessage.warning('请先确认或取消当前标注');return}
  if(videoWorkflow.value==='first_frame_auto')seekVideoFrame(0)
}
async function chooseCategory(label){
  if(videoLocked.value)return
  if(label===null){
    try{
      const response=await ElMessageBox.prompt('输入需要的新类别，保存后仅你自己可以复用。','自定义类别',{inputPlaceholder:'类别名称',inputValidator:value=>!!value?.trim() && value.trim().length<=100 || '请输入1至100个字符',confirmButtonText:'添加',cancelButtonText:'取消'})
      label=response.value.trim()
      await api.post('/video-annotation/labels',{name:label})
      if(!videoClasses.value.includes(label))videoClasses.value.push(label)
    }catch{return}
  }
  currentLabel.value=label
  if(videoDraft.value)videoDraft.value.label=label
  redraw()
}
function cancelVideoDraft(){
  if(videoLocked.value)return
  invalidateSeeds();videoDraft.value=null;editingVideoKey.value=null;videoEditMode.value='box';redraw()
}
async function confirmVideoDraft(){
  if(videoLocked.value || seedPending.value || !draftReady.value)return
  pauseVideo();const draft=clone(videoDraft.value);submitting.value=true
  try{
    await api.post('/video-annotation/labels',{name:draft.label})
    delete draft.seed_status;delete draft.seed_error
    draft.source='manual';draft.review_status='human_edited';draft.manual_confirmed=true
    if(videoWorkflow.value==='first_frame_auto')draft.generation={status:'prepared',output_modes:[...outputTypes.value],end_frame:autoEndFrame.value}
    else if(draft.frame===0 && draft.generation)draft.generation={...draft.generation,status:'stale'}
    const key=editingVideoKey.value || annotationKey(draft)
    const index=anns.value.findIndex(a=>a.type!=='video_point' && annotationKey(a)===key)
    if(index>=0)anns.value[index]=draft;else anns.value.push(draft)
    // A single confirmed edit changes one frame only, including its point labels.
    for(const a of anns.value)if(a.type==='video_point' && annotationKey(a)===key)a.label=draft.label
    const hi=manualVideoResults.value.findIndex(a=>annotationKey(a)===key)
    if(hi>=0)manualVideoResults.value[hi]=clone(draft);else manualVideoResults.value.push(clone(draft))
    if(!videoClasses.value.includes(draft.label))videoClasses.value.push(draft.label)
    invalidateSeeds();videoDraft.value=null;editingVideoKey.value=null;videoEditMode.value='box';redraw()
  }catch{/* preserve the draft on save failure */}finally{submitting.value=false}
}
function previewVideoAnnotation(a){
  if(videoLocked.value || videoDraft.value)return
  pauseVideo()
  if(videoWorkflow.value==='first_frame_auto'){
    if(!objectResultReady(a))return
    objectPreviewId.value=a.track_id;videoPreview.value=true;previewedVideoAnnotation.value=null;seekVideoFrame(0)
  }else{seekVideoFrame(a.frame);previewedVideoAnnotation.value=a}
  redraw()
}
function editVideoAnnotation(a){
  if(videoLocked.value)return
  if(videoDraft.value){ElMessage.warning('请先确认或取消当前标注');return}
  pauseVideo();previewedVideoAnnotation.value=null;objectPreviewId.value=null;videoPreview.value=false
  if(videoWorkflow.value==='first_frame_auto' && a.frame!==0)videoWorkflow.value='manual'
  seekVideoFrame(a.frame)
  const copy=clone(a)
  if(!copy.tracking_points?.length){
    copy.tracking_points=anns.value.filter(p=>p.type==='video_point' && annotationKey(p)===annotationKey(a) && p.visible!==false && p.xy).map(p=>({point_id:p.point_id,xy:p.xy,origin:p.seed_origin||'manual'}))
  }
  copy.points_dirty=!!anns.value.find(b=>b.type!=='video_point' && annotationKey(b)===annotationKey(a))?.points_dirty;copy.points_manual=!!copy.tracking_points?.length
  editingVideoKey.value=annotationKey(a);videoDraft.value=copy;currentLabel.value=copy.label;videoEditMode.value='box'
  if(!copy.mask)queueSeed(videoDraft.value)
  else if(outputTypes.value.includes('trajectory') && !copy.tracking_points.length)resamplePoints(videoDraft.value)
  redraw()
}
async function deleteVideoAnnotation(a){
  if(videoLocked.value || videoDraft.value)return
  const key=annotationKey(a),wholeObject=videoWorkflow.value==='first_frame_auto'
  try{await ElMessageBox.confirm(wholeObject?`删除「${a.label}」的首帧标注及后续自动结果？其他物体保留，保存后生效。`:`删除第 ${a.frame} 帧的「${a.label}」及其当前帧的点？其他条目和帧保留，保存后生效。`,'删除标注',{confirmButtonText:'删除',cancelButtonText:'取消',type:'warning'})}catch{return}
  anns.value=anns.value.filter(row=>wholeObject?row.track_id!==a.track_id:annotationKey(row)!==key)
  manualVideoResults.value=manualVideoResults.value.filter(row=>wholeObject?row.track_id!==a.track_id:annotationKey(row)!==key)
  if(previewedVideoAnnotation.value && annotationKey(previewedVideoAnnotation.value)===key)previewedVideoAnnotation.value=null
  redraw()
}
function resamplePoints(a){
  if(!a?.mask || videoLocked.value)return
  a.tracking_points=sampleMaskPoints(a.mask,a.point_limit||8);a.points_manual=false;a.points_dirty=true;redraw()
}
function outputsChanged(){
  const a=videoDraft.value;if(outputTypes.value.includes('trajectory') && a?.mask && !a.tracking_points?.length)resamplePoints(a)
  redraw()
}
function setPointLimit(value){
  const a=selectedAnn.value;if(!a || videoLocked.value)return
  if(a.points_manual && (a.tracking_points?.length||0)>value)return ElMessage.warning('请先删除多余手动点')
  a.point_limit=value;if(!a.points_manual)resamplePoints(a)
}
function addTrackingPoint(e){
  const a=selectedAnn.value;if(!a?.xyxy)return
  const [cx,cy]=canvasPos(e),xy=[cx*natW/cvEl.value.width,cy*natH/cvEl.value.height]
  a.tracking_points||=[]
  const near=a.tracking_points.findIndex(p=>Math.hypot((p.xy[0]-xy[0])*cvEl.value.width/natW,(p.xy[1]-xy[1])*cvEl.value.height/natH)<8)
  if(near>=0){a.tracking_points.splice(near,1);a.points_manual=true;a.points_dirty=true;redraw();return}
  if(!(0<=xy[0]&&xy[0]<natW&&0<=xy[1]&&xy[1]<natH))return ElMessage.warning('请在视频画面内选点')
  if(a.tracking_points.length>=(a.point_limit||8))return ElMessage.warning('已达到此物体选点上限；点击已有点可删除')
  let pid=1;while(a.tracking_points.some(p=>p.point_id===pid))pid++
  a.tracking_points.push({point_id:pid,xy:xy.map(v=>Math.round(v*100)/100),origin:'manual'});a.points_manual=true;a.points_dirty=true;redraw()
}
// One queued draft request at a time; revisions and context prevent stale mask writes.
let seedQueue=[],seedRunning=false,seedEpoch=0,seedTask=null
function queueSeed(a){
  a.seed_status='queued';delete a.seed_error
  seedQueue=seedQueue.filter(x=>x.a!==a);seedQueue.push({a,epoch:seedEpoch,box:JSON.stringify(a.xyxy),frame:a.frame})
  runSeedQueue()
}
async function runSeedQueue(){
  if(seedRunning)return
  seedRunning=true
  try{while(seedQueue.length){
    const item=seedQueue.shift(),a=item.a
    const valid=()=>item.epoch===seedEpoch && videoDraft.value===a && JSON.stringify(a.xyxy)===item.box
    if(!valid())continue
    a.seed_status='running'
    try{
      let job=await api.post('/video-annotation',{dataset_id:datasetId.value,input_version_id:inputVersionId.value,file:keyOf(currentFile.value),
        start_frame:item.frame,frame_count:1,workflow:videoWorkflow.value,purpose:'seed',output_modes:outputTypes.value.length?[...outputTypes.value]:['box'],
        prompts:[{track_id:a.track_id,label:a.label,xyxy:a.xyxy,point_limit:a.point_limit||8}],base_label:[]})
      seedTask=job.id
      if(!valid()){await api.post(`/video-annotation/${job.id}/cancel`).catch(()=>{});continue}
      while(['pending','running'].includes(job.status)){
        await new Promise(resolve=>setTimeout(resolve,700))
        if(!valid()){await api.post(`/video-annotation/${job.id}/cancel`).catch(()=>{});break}
        job=await api.get(`/video-annotation/${job.id}`)
      }
      if(!valid())continue
      if(job.status!=='completed')throw Error(job.error_msg||'mask任务已取消')
      const result=await api.get(`/video-annotation/${job.id}/result`)
      if(!valid())continue
      const row=result.annotations.find(x=>x.track_id===a.track_id)
      if(!row?.mask)throw Error('SAM未识别出有效区域，请重画框')
      a.prompt_xyxy=JSON.parse(item.box);a.mask=row.mask
      a.xyxy=row.xyxy || canvasMask(maskCanvas(row.mask)).xyxy;a.box_edited=false;a.seed_status='ready'
      if(outputTypes.value.includes('trajectory') && !a.points_manual)resamplePoints(a)
      redraw()
    }catch(e){if(valid()){a.seed_status='failed';a.seed_error=e.friendlyMessage||e.message||'mask生成失败'}}
    finally{seedTask=null}
  }}finally{seedRunning=false}
}
function invalidateSeeds(){seedEpoch++;seedQueue=[];if(seedTask)api.post(`/video-annotation/${seedTask}/cancel`).catch(()=>{})}
// Model requests and result-version loading; polling lives in VideoAssist.
async function generateConfirmed(){
  if(videoLocked.value || videoDraft.value)return
  const objects=humanResults.value
  if(!objects.length || objects.length>10)return ElMessage.warning('请确认1至10个首帧物体')
  if(!outputTypes.value.length)return ElMessage.warning('请至少选择一种输出文件')
  if(objects.some(a=>!hasMaskPixels(a.mask)))return ElMessage.warning('请补齐各物体的有效 mask')
  if(outputTypes.value.includes('trajectory') && objects.some(a=>!a.tracking_points?.length))return ElMessage.warning('请补齐各物体的轨迹点')
  activeObjectKeys.value=objects.map(annotationKey)
  for(const a of objects)a.generation={status:'prepared',output_modes:[...outputTypes.value],end_frame:autoEndFrame.value}
  const prompts=objects.map(a=>({track_id:a.track_id,label:a.label,xyxy:a.xyxy,mask:a.mask,points:a.tracking_points||[],point_limit:a.point_limit||8}))
  backgroundSubmitting.value=true
  try{
    const job=await videoAssist.value.run({dataset_id:datasetId.value,input_version_id:inputVersionId.value,file:keyOf(currentFile.value),start_frame:0,frame_count:autoEndFrame.value+1,
      workflow:'first_frame_auto',output_modes:[...outputTypes.value],prompts,base_label:encodedVideoLabel.value})
    if(!job){for(const a of objects)a.generation={...a.generation,status:'failed',error:'任务提交失败，请重试'};activeObjectKeys.value=[]}
  }finally{backgroundSubmitting.value=false}
}
function updateObjectJob(job){
  for(const a of manualVideoResults.value.filter(a=>activeObjectKeys.value.includes(annotationKey(a)))){
    a.generation.status=job.status==='completed'?'running':job.status
    a.generation.task_id=job.id;a.generation.error=job.error_msg||''
  }
  // Accepted inputs are persisted by the backend; polling changes are not unsaved edits.
  if(modelSnapshot.value)openedManualSnapshot.value=JSON.stringify(manualVideoResults.value)
}
async function completeObjectGeneration(job){
  objectLoading.value=true
  const objects=manualVideoResults.value.filter(a=>a.generation?.task_id===job.id)
  try{
    const raw=await api.get(`/datasets/${job.dataset_id}/versions/${job.output_version_id}/file`,{params:{path:labelKeyOf(job.file)}})
    const state=decodeLabel(raw,'video'),key=keyOf(currentFile.value)
    anns.value=state.items;manualVideoResults.value=clone(raw.manual_annotations||[]);adapters.value[key]=state;labelsMap.value[key]=state.items
    openedSnapshot.value=JSON.stringify(anns.value);openedManualSnapshot.value=JSON.stringify(manualVideoResults.value)
    inputVersionId.value=job.output_version_id;loadedContext=null;changed.value=new Set();fileUrl.value=fileUrlOf(key)
    activeObjectKeys.value=[];videoPreview.value=false;objectPreviewId.value=null;initialPreviewFrame=0
    await loadDatasets();await loadFiles(1);videoTasks.value?.reload();await nextTick();syncVideoCanvas()
    ElMessage.success('后台自动标注完成，可预览各物体或在任务列表查看完整结果')
  }catch(e){for(const a of objects)a.generation={...a.generation,status:'failed',error:'结果读取失败，请从任务列表打开已完成结果'};ElMessage.error(e.friendlyMessage||'结果读取失败')}
  finally{objectLoading.value=false;videoBusy.value=false}
}
async function finishVideo(){
  pauseVideo()
  if(videoDraft.value){ElMessage.warning('请先确认或取消当前标注');return}
  if(!outputTypes.value.length)return ElMessage.warning('请至少选择一种输出文件')
  return saveVideoFile()
}
async function downloadVideoLabels(){
  try{
    const blob=await api.get('/video-annotation/exports',{params:{dataset_id:datasetId.value,input_version_id:inputVersionId.value,file:keyOf(currentFile.value)},responseType:'blob'})
    const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='video_annotations.zip';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)
  }catch{/* shared API feedback */}
}
function selectAnnotation(i){selectedIdx.value=i;redraw()}
function setVideoTool(mode){pauseVideo();videoEditMode.value=mode;redraw()}
watch(videoFrame,()=>{drag=null;if(previewedVideoAnnotation.value?.frame!==videoFrame.value)previewedVideoAnnotation.value=null}, {flush:'sync'})
async function closeWorkbench(done){
  if(submitting.value || backgroundSubmitting.value || objectLoading.value)return
  pauseVideo()
  if(workbenchDirty.value){try{await ElMessageBox.confirm('当前修改尚未保存，关闭将放弃这些修改。是否继续？','未保存修改')}catch{return}}
  invalidateSeeds();videoDraft.value=null;editingVideoKey.value=null;if(typeof done==='function')done();else wbVisible.value=false
}
onBeforeRouteLeave(async()=>{
  if(submitting.value || backgroundSubmitting.value || objectLoading.value)return false
  if((wbVisible.value&&workbenchDirty.value)||changed.value.size){
    try{await ElMessageBox.confirm('离开将放弃未保存或暂存的标注，是否继续？','未保存修改')}catch{return false}
  }
})
async function saveVideoFile(){
  if(videoLocked.value || videoDraft.value)return
  pauseVideo();submitting.value=true
  try{
    const task=await api.post('/manual-label',{dataset_id:datasetId.value,input_version_id:inputVersionId.value,label_type:labelType.value,
      labels:[{file:keyOf(currentFile.value),label:encodedVideoLabel.value}]})
    openedSnapshot.value=JSON.stringify(anns.value);wbVisible.value=false
    changed.value=new Set();labelsMap.value={};adapters.value={}
    inputVersionId.value=task.output_version_id
    ElMessage.success('已保存为新版本，无需再次提交')
    const row={...currentFile.value,has_label:true},frame=videoFrame.value
    await loadDatasets();await loadFiles(1);manualTasks.value?.reload()
    await openWorkbench(row);videoPreview.value=true;initialPreviewFrame=frame;submitting.value=false;await nextTick();syncVideoCanvas()
  }catch{/* Keep working edits on API failure. */}finally{submitting.value=false}
}
let videoCallback=null, callbackVideo=null
function stopVideoCallback(){if(callbackVideo&&videoCallback!=null)callbackVideo.cancelVideoFrameCallback?.(videoCallback);videoCallback=null}
function animateVideo(){
  stopVideoCallback();const v=videoEl.value;callbackVideo=v
  if(!v?.requestVideoFrameCallback)return
  const tick=(_,meta)=>{if(v!==videoEl.value||!wbVisible.value||v.paused)return;redraw(meta.mediaTime);videoCallback=v.requestVideoFrameCallback(tick)}
  videoCallback=v.requestVideoFrameCallback(tick)
}
onBeforeUnmount(()=>{stopVideoCallback();invalidateSeeds()})
async function cancelVideoTask(row) { await api.post(`/video-annotation/${row.id}/cancel`);videoTasks.value?.reload() }
async function openVideoResult(job) {
  if (changed.value.size || (wbVisible.value && JSON.stringify(anns.value)!==(modelSnapshot.value || openedSnapshot.value))) {
    try {await ElMessageBox.confirm('打开结果将替换当前工作台；未提交的本地修改将放弃，是否继续？','打开结果')}catch{return}
  }
  wbVisible.value=false;changed.value=new Set();labelsMap.value={};adapters.value={};loadedContext=null
  await loadDatasets();datasetId.value=job.dataset_id;inputVersionId.value=job.output_version_id;await loadFiles(1)
  await openWorkbench({name:job.file.split('/').pop(),path:job.file,has_label:true})
  outputTypes.value=job.output_modes?.length?job.output_modes:(adapters.value[job.file]?.original?.export_types || [job.output_mode || 'mask']);videoPreview.value=true;initialPreviewFrame=job.start_frame || 0
  await nextTick();syncVideoCanvas()
}
async function openVideoInputs(job){
  if(submitting.value || backgroundSubmitting.value || objectLoading.value)return
  if(wbVisible.value || changed.value.size){ElMessage.warning('请先关闭当前工作台并保存修改');return}
  try{
    const saved=await api.get(`/video-annotation/${job.id}/input`)
    labelsMap.value={};adapters.value={};loadedContext=null;await loadDatasets()
    datasetId.value=job.dataset_id;inputVersionId.value=job.input_version_id;await loadFiles(1)
    await openWorkbench({name:job.file.split('/').pop(),path:job.file,has_label:true})
    if(!wbVisible.value)return
    const state=decodeLabel(saved.base_label,'video'),key=keyOf(currentFile.value)
    anns.value=state.items;manualVideoResults.value=clone(saved.base_label.manual_annotations || [])
    for(const a of manualVideoResults.value)a.generation={status:'prepared'}
    adapters.value[key]=state;labelsMap.value[key]=state.items
    outputTypes.value=saved.output_modes;videoWorkflow.value='first_frame_auto';autoEndFrame.value=saved.frame_count-1
    openedSnapshot.value=JSON.stringify(anns.value);openedManualSnapshot.value=JSON.stringify(manualVideoResults.value)
    await nextTick();syncVideoCanvas();ElMessage.success('已恢复该任务提交时的首帧，可修正后重新启动后台标注')
  }catch(e){ElMessage.error(e.friendlyMessage||'无法恢复任务输入')}
}
function seekVideoFrame(frame) {
  const v=videoEl.value;if(!v || videoLocked.value)return
  if(videoDraft.value){ElMessage.warning('请先确认或取消当前标注');return}
  onBoxEnd()
  frame=Math.max(0,Math.min((videoMeta.value.frame_count||1)-1,Math.round(frame)))
  const times=adapters.value[keyOf(currentFile.value || {})]?.original?.frame_times || []
  v.pause();v.currentTime=times.find(t=>t.frame===frame)?.time ?? frame/videoFps();videoFrame.value=frame
}
async function openWorkbench(row) {
  invalidateSeeds();objectPreviewId.value=null;activeObjectKeys.value=[];maskStroke=null;previewedVideoAnnotation.value=null;videoDraft.value=null;editingVideoKey.value=null;manualVideoResults.value=[]
  currentFile.value = row
  stopVideoCallback();videoBusy.value=false;videoPreview.value=false;initialPreviewFrame=null;modelSnapshot.value=null;maskCache.clear();videoEditMode.value='box';videoFrame.value=0
  if(modality.value==='video') {
    try {
      const [meta,caps,labels]=await Promise.all([api.get('/video-annotation/metadata',{params:{dataset_id:datasetId.value,input_version_id:inputVersionId.value,file:keyOf(row)}}),api.get('/video-annotation/capabilities'),api.get('/video-annotation/labels')])
      videoMeta.value=meta;videoClasses.value=[...new Set([...caps.classes,...labels.items])];autoEndFrame.value=Math.min(299,meta.frame_count-1)
    }catch{return}
  }
  const key = keyOf(row)
  // 懒加载该文件已有标注（本地未缓存时从版本目录取 _label.json）
  if (!labelsMap.value[key]) {
    try {
      let text = ''
      if (modality.value === 'text') {
        const response = await fetch(fileUrlOf(key), { credentials: 'include' })
        const bytes = await response.arrayBuffer()
        try { text = new TextDecoder('utf-8', { fatal: true }).decode(bytes) }
        catch { text = new TextDecoder('gb18030', { fatal: true }).decode(bytes) }
      }
      let raw = []
      if (row.has_label) {
        const res = await fetch(fileUrlOf(labelKeyOf(key)), { credentials: 'include' })
        if (!res.ok) throw new Error('原标注读取失败，已阻止覆盖')
        raw = await res.json()
      }
      const state = decodeLabel(raw, modality.value, text)
      adapters.value[key] = state
      labelsMap.value[key] = state.items
    } catch (e) { ElMessage.error(e.message || '原标注无法解析'); return }
  }
  anns.value = clone(labelsMap.value[key] || [])
  currentLabel.value = ''
  if(modality.value==='video') {let next=Math.max(0,...anns.value.map(a=>a.track_id||0))+1;for(const a of anns.value)if(a.xyxy&&!a.track_id)a.track_id=next++}
  if(modality.value==='video'){
    const original=adapters.value[key]?.original
    outputTypes.value=original?.export_types || ['box'];videoWorkflow.value=original?.workflow || 'manual'
    manualVideoResults.value=clone(original?.manual_annotations || anns.value.filter(a=>a.xyxy && a.source==='manual'))
    openedManualSnapshot.value=JSON.stringify(manualVideoResults.value)
    autoEndFrame.value=Math.min(videoMeta.value.frame_count-1,manualVideoResults.value.find(a=>a.generation?.end_frame!=null)?.generation.end_frame ?? autoEndFrame.value)
    // Reopen tracked points as editable per-object seeds for manual corrections.
    const pointsByObjectFrame=new Map()
    for(const a of anns.value)if(a.type==='video_point' && a.visible!==false && a.xy){
      const key=`${a.track_id}:${a.frame}`;if(!pointsByObjectFrame.has(key))pointsByObjectFrame.set(key,[]);pointsByObjectFrame.get(key).push(a)
    }
    for(const a of anns.value)if(a.xyxy){
      const points=pointsByObjectFrame.get(`${a.track_id}:${a.frame}`)||[]
      if(points.length){a.tracking_points=points.map(b=>({point_id:b.point_id,xy:b.xy,origin:b.seed_origin||'manual'}));a.points_manual=true}
      a.points_dirty=false
    }
  }
  openedSnapshot.value=JSON.stringify(anns.value)
  selectedIdx.value = -1
  fileUrl.value = fileUrlOf(key)
  wbVisible.value = true
  wbLoading.value = true
  try {
    if (modality.value === 'voice') await loadWave()
    else if (modality.value === 'radar') await loadBev()
    else if (modality.value === 'text') await loadText()
    else if (modality.value === 'image') await waitImgReady()
  } finally {
    wbLoading.value = false
    await nextTick()
    if (modality.value === 'radar' && bevEl.value) {
      bevEl.value.width = bevEl.value.clientWidth || 900
      bevEl.value.height = 520
      cvEl.value = bevEl.value
    }
    redraw()
  }
}

// 图像模态：等待 <img> 真正加载完成后再允许绘制（否则画布按错误尺寸建立）
function waitImgReady() {
  return new Promise((resolve) => {
    nextTick(() => {
      const img = imgEl.value
      if (!img) return resolve()
      if (img.complete && img.naturalWidth > 0) return resolve()
      const timer = setTimeout(resolve, 10000)  // 兜底 10s
      img.addEventListener('load', () => { clearTimeout(timer); resolve() }, { once: true })
      img.addEventListener('error', () => { clearTimeout(timer); resolve() }, { once: true })
    })
  })
}

function annDesc(a) {
  const lb = a.label || '未命名'
  if (a.type === 'point') return `${lb} (${a.center?.join(', ')})`
  if (a.type === 'segment') return `${lb} [${a.start.toFixed(1)}s-${a.end.toFixed(1)}s]`
  if (a.type === 'span') return `${lb} "${(a.text || '').slice(0, 12)}"`
  if (a.type === 'bev_box') return `${lb} R[${a.range.map((v) => v.toFixed(0))}]`
  return lb
}

function removeSelected() {
  if (selectedIdx.value >= 0) {
    anns.value.splice(selectedIdx.value, 1)
    selectedIdx.value = -1
    redraw()
  }
}

// 自动标注产生的框带 confidence 字段（格式适配器保留），据此区分来源
const isAutoAnn = a => a && a.source !== 'manual' && (a.confidence != null || a.source === 'sam2.1')
const autoCount = computed(() => anns.value.filter(isAutoAnn).length)

// 一键清除本文件全部自动标注框（保留手动框），用于批量剔除误检
function clearAuto() {
  anns.value = anns.value.filter((a) => !isAutoAnn(a))
  selectedIdx.value = -1
  redraw()
  ElMessage.success('已清除自动标注框，手动标注保留；记得保存本文件')
}

function clearAll() {
  anns.value = []
  selectedIdx.value = -1
  redraw()
}

function saveFile() {
  const key = keyOf(currentFile.value)
  labelsMap.value[key] = clone(anns.value)
  changed.value.add(key)
  wbVisible.value = false
  ElMessage.success('已暂存，记得点右上角"提交标注"')
}

async function onSubmit() {
  if (!changed.value.size) return ElMessage.warning('暂无改动，请先在画布中标注')
  submitting.value = true
  try {
    const labels = [...changed.value].map((f) => ({ file: f, label: encodeLabel(adapters.value[f], labelsMap.value[f] || []) }))
    await api.post('/manual-label', {
      dataset_id: datasetId.value, label_type: labelType.value,
      labels, input_version_id: inputVersionId.value,
    })
    ElMessage.success('标注已提交并生成新版本')
    changed.value = new Set()
    await loadDatasets()
    labelsMap.value = {}; adapters.value = {}
    inputVersionId.value = currentVersionId(datasetId.value)
    await loadFiles(1)
  } catch { /* 已提示 */ } finally { submitting.value = false }
}

// ---------- 图像 / 视频 / 雷达：拖框 ----------
const stageRef = ref(null)
const imgEl = ref(null)
const videoEl = ref(null)
const cvEl = ref(null)
let videoResizeObserver=null
watch(videoEl,v=>{videoResizeObserver?.disconnect();if(v){videoResizeObserver=new ResizeObserver(syncVideoCanvas);videoResizeObserver.observe(v)}})
onBeforeUnmount(()=>videoResizeObserver?.disconnect())
let natW = 1, natH = 1      // 原始坐标系尺寸
let drag = null             // {x1,y1,x2,y2} 画布坐标

function onImgLoad(e) {
  natW = e.target.naturalWidth
  natH = e.target.naturalHeight
  nextTick(syncCanvas)
}
function syncVideoCanvas() {
  if(initialPreviewFrame!=null && videoEl.value?.readyState>=1){const frame=initialPreviewFrame;initialPreviewFrame=null;seekVideoFrame(frame)}
  syncCanvas()
}
function syncCanvas() {
  const cv = cvEl.value
  const media = modality.value === 'video' ? videoEl.value : imgEl.value
  if (!cv || !media) return
  cv.width = media.clientWidth
  cv.height = media.clientHeight
  if (modality.value === 'video') {
    natW = videoEl.value.videoWidth || media.clientWidth
    natH = videoEl.value.videoHeight || media.clientHeight
  }
  redraw()
}
function canvasPos(e) {
  const r = cvEl.value.getBoundingClientRect()
  return [Math.max(0,Math.min(cvEl.value.width,e.clientX-r.left)),Math.max(0,Math.min(cvEl.value.height,e.clientY-r.top))]
}
function startMaskStroke(e){
  const a=videoDraft.value
  if(!a?.mask || seedPending.value || videoLocked.value)return
  const canvas=maskCanvas(a.mask);maskStroke={a,canvas,last:null,erase:videoEditMode.value==='erase'}
  showMask.value=true;paintMask(e)
}
function paintMask(e){
  const stroke=maskStroke;if(!stroke)return
  const [x,y]=canvasPos(e),p=[x*natW/cvEl.value.width,y*natH/cvEl.value.height]
  paintMaskStroke(stroke.canvas,stroke.last,p,maskBrushSize.value,stroke.erase)
  stroke.last=p;redraw()
}
function finishMaskStroke(){
  const stroke=maskStroke;maskStroke=null
  if(!stroke || videoDraft.value!==stroke.a)return
  const a=stroke.a,{mask,xyxy}=canvasMask(stroke.canvas);a.mask=mask;a.mask_edited=true
  if(xyxy && !a.box_edited)a.xyxy=xyxy
  if(outputTypes.value.includes('trajectory')){
    if(!a.points_manual)resamplePoints(a)
    else{
      a.tracking_points=pointsInMask(stroke.canvas,a.tracking_points||[]);a.points_dirty=true
    }
  }
  redraw()
}
function onBoxStart(e) {
  if(modality.value==='video'){
    if(videoLocked.value || videoPreview.value)return
    pauseVideo()
    if(videoWorkflow.value==='first_frame_auto' && videoFrame.value!==0)return ElMessage.warning('自动模式请在首帧框选物体；后续修正请选择逐帧手动模式')
    if(videoEditMode.value==='point'){addTrackingPoint(e);return}
    if(['brush','erase'].includes(videoEditMode.value)){startMaskStroke(e);return}
    if(!currentLabel.value)return ElMessage.warning('请先选择或自定义类别')
    previewedVideoAnnotation.value=null
  }
  const [x, y] = canvasPos(e)
  // 命中已有框则选中
  const sx = cvEl.value.width / natW, sy = cvEl.value.height / natH
  for (let i = anns.value.length - 1; i >= 0; i--) {
    const a = anns.value[i]
    if (!visibleAtCurrentFrame(a)) continue
    if (a.xyxy && modality.value!=='video') {
      const [x1, y1, x2, y2] = a.xyxy
      if (x >= x1 * sx && x <= x2 * sx && y >= y1 * sy && y <= y2 * sy) { selectedIdx.value = i; redraw(); return }
    }
  }
  drag = { x1: x, y1: y, x2: x, y2: y }
}
function onBoxMove(e) {
  if(maskStroke){paintMask(e);return}
  if (!drag) return
  const [x, y] = canvasPos(e)
  drag.x2 = x; drag.y2 = y
  redraw()
}
function onBoxEnd() {
  if(maskStroke){finishMaskStroke();return}
  if (!drag) return
  const { x1, y1, x2, y2 } = drag
  drag = null
  if (Math.abs(x2 - x1) < 5 || Math.abs(y2 - y1) < 5) { redraw(); return }
  const sx = natW / cvEl.value.width, sy = natH / cvEl.value.height
  const xyxy = [Math.min(x1, x2) * sx, Math.min(y1, y2) * sy, Math.max(x1, x2) * sx, Math.max(y1, y2) * sy]
    .map((v) => Math.round(v * 10) / 10)
  if (modality.value === 'radar') {
    // 画布坐标 → 数据坐标（BEV）
    const d = bevData
    const rx = (v) => d.xMin + (v / cvEl.value.width) * (d.xMax - d.xMin)
    const ry = (v) => d.yMax - (v / cvEl.value.height) * (d.yMax - d.yMin)
    anns.value.push({ type: 'bev_box', range: [rx(Math.min(x1, x2)), rx(Math.max(x1, x2))],
                      azimuth: [ry(Math.max(y1, y2)), ry(Math.min(y1, y2))],
                      label: currentLabel.value || '目标' })
  } else if (modality.value === 'video') {
    let a=videoDraft.value
    if(a?.mask){a.xyxy=xyxy;a.box_edited=true;a.outside=false;redraw();return}
    if(a){a.xyxy=xyxy;a.outside=false}
    else{
      const trackId=anns.value.reduce((n,a)=>Math.max(n,a.track_id||0),0)+1
      videoDraft.value={type:'video_box',time:videoFrame.value/videoFps(),frame:videoFrame.value,xyxy,
        track_id:trackId,label:currentLabel.value,point_limit:8,points_dirty:true,source:'manual',review_status:'human_edited'}
      a=videoDraft.value
    }
    queueSeed(a);redraw();return
  } else {
    anns.value.push({ type: 'bbox', xyxy, label: currentLabel.value || '目标' })
  }
  selectedIdx.value = anns.value.length - 1
  redraw()
}

function videoFps() { return adapters.value[keyOf(currentFile.value || {})]?.original?.fps || videoMeta.value.fps || 25 }
function visibleAtCurrentFrame(a) {
  return modality.value !== 'video' || (a.frame ?? Math.round((a.time || 0) * videoFps())) === videoFrame.value
}
async function toggleVideo() {
  const video = videoEl.value
  if (!video || videoTimelineLocked.value) return
  onBoxEnd()
  try { if (video.paused) await video.play(); else video.pause() }
  catch { ElMessage.error("浏览器无法播放此编码，请使用 H.264 MP4 视频进行在线标注") }
}
function stepFrame(delta) {
  const video = videoEl.value
  if(video)seekVideoFrame(Math.max(0,Math.min((videoMeta.value.frame_count||1)-1,videoFrame.value+delta)))
}
function redraw(mediaTime = null) {
  if(modality.value==='video' && videoEl.value) {
    const time=typeof mediaTime==='number'?mediaTime:videoEl.value.currentTime,times=adapters.value[keyOf(currentFile.value || {})]?.original?.frame_times || []
    let frame=Math.round(time*videoFps())
    for(const t of times)if(Math.abs(t.time-time)<.5/videoFps()){frame=t.frame;break}
    videoFrame.value=Math.max(0,Math.min((videoMeta.value.frame_count||1)-1,frame))
  }
  const cv = cvEl.value
  if (cv && (modality.value === 'image' || modality.value === 'video')) {
    const ctx = cv.getContext('2d')
    ctx.clearRect(0, 0, cv.width, cv.height)
    const sx = cv.width / natW, sy = cv.height / natH
    if(modality.value==='video' && (videoPreview.value?previewTypes.value:outputTypes.value).includes('trajectory'))drawTrajectories(ctx,sx,sy)
    const displayIndices=modality.value==='video'?new Set(visibleAnnotations.value.map(({i})=>i)):null
    const overlay=modality.value==='video' ? (videoDraft.value || previewedVideoAnnotation.value) : null
    const renderRows=overlay ? [...anns.value,overlay] : anns.value
    renderRows.forEach((a, i) => {
      const draft=modality.value==='video' && a===videoDraft.value
      const highlighted=!!overlay && a===overlay
      if(!highlighted && overlay && annotationKey(a)===annotationKey(overlay))return
      if(!highlighted && displayIndices&&!displayIndices.has(i))return
      if(modality.value==='video' && a.type==='video_point')return
      if (a.outside) return
      if (!a.xyxy && !a.mask) return
      if (!visibleAtCurrentFrame(a)) return
      if(a.mask && (modality.value==='video' && videoPreview.value ? previewTypes.value.includes('mask') : showMask.value)) {
        try {
          let canvas=maskStroke?.a===a ? maskStroke.canvas : maskCache.get(a.mask)
          if(!canvas){canvas=maskCanvas(a.mask);maskCache.set(a.mask,canvas);if(maskCache.size>8)maskCache.delete(maskCache.keys().next().value)}
          if(maskStroke?.a===a)ctx.globalAlpha=.4
          ctx.drawImage(canvas,0,0,cv.width,cv.height);ctx.globalAlpha=1
        }catch{ /* malformed external mask is preserved, not drawn */ }
      }
      if(!a.xyxy || (modality.value==='video'&&videoPreview.value&&!previewTypes.value.includes('box')))return
      const [x1, y1, x2, y2] = a.xyxy
      ctx.strokeStyle = (modality.value==='video' ? draft : i===selectedIdx.value) ? '#ef4444' : (highlighted ? '#409eff' : '#fbbf24')
      ctx.lineWidth = 2
      ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy)
      drawTag(ctx,a.label,x1*sx,y1*sy)
      if(modality.value==='video' && outputTypes.value.includes('trajectory') && !videoPreview.value){
        const points=a.tracking_points||[]
        for(const p of points)drawPoint(ctx,p.xy,sx,sy,`点${p.point_id}`,'#38bdf8')
      }
    })
    if (drag) {
      ctx.strokeStyle = '#34d399'
      ctx.setLineDash([4, 4])
      ctx.strokeRect(drag.x1, drag.y1, drag.x2 - drag.x1, drag.y2 - drag.y1)
      ctx.setLineDash([])
    }
  }
  if (modality.value === 'voice') drawWave()
  if (modality.value === 'radar') drawBev()
}
function drawPoint(ctx,xy,sx,sy,label,color){
  if(!xy)return
  ctx.fillStyle=color;ctx.beginPath();ctx.arc(xy[0]*sx,xy[1]*sy,4,0,2*Math.PI);ctx.fill();drawTag(ctx,label,xy[0]*sx+6,xy[1]*sy)
}
const trajectoryGroups=computed(()=>{
  const groups=new Map()
  for(const a of anns.value)if(a.type==='video_point'){const k=`${a.track_id}.${a.point_id}`;if(!groups.has(k))groups.set(k,[]);groups.get(k).push(a)}
  for(const rows of groups.values())rows.sort((a,b)=>a.frame-b.frame)
  return groups
})
function drawTrajectories(ctx,sx,sy){
  for(const rows of trajectoryGroups.value.values()){
    const current=rows.find(a=>a.frame===videoFrame.value);if(!current)continue
    if(objectPreviewId.value!==null && (current.track_id!==objectPreviewId.value || current.frame>autoEndFrame.value))continue
    if(!videoPreview.value && videoDraft.value && annotationKey(videoDraft.value)===annotationKey(current))continue
    ctx.strokeStyle=`hsl(${(current.track_id*83+current.point_id*31)%360} 90% 65%)`;ctx.lineWidth=2
    let prev=null;ctx.beginPath()
    for(const a of rows){
      if(a.frame>videoFrame.value)break
      if(!a.xy || a.visible===false || a.outside){prev=null;continue}
      if(prev && a.frame===prev.frame+1 && a.trajectory_run===prev.trajectory_run)ctx.lineTo(a.xy[0]*sx,a.xy[1]*sy);else ctx.moveTo(a.xy[0]*sx,a.xy[1]*sy)
      prev=a
    }
    ctx.stroke()
    if(current.visible!==false && !current.outside)drawPoint(ctx,current.xy,sx,sy,`点${current.point_id}`,ctx.strokeStyle)
  }
}
function drawTag(ctx, text, x, y) {
  if (!text) return
  ctx.font = '12px sans-serif'
  const w = ctx.measureText(text).width + 8
  ctx.fillStyle = 'rgba(251,191,36,0.9)'
  ctx.fillRect(x, Math.max(y - 18, 0), w, 16)
  ctx.fillStyle = '#0c1220'
  ctx.fillText(text, x + 4, Math.max(y - 5, 12))
}

// ---------- 语音：波形 + 段选 ----------
const waveEl = ref(null)
let waveData = null          // {samples: Float32Array, duration}
let segDrag = null           // {t1, t2}

async function loadWave() {
  const res = await fetch(fileUrl.value, { credentials: 'include' })
  const buf = await res.arrayBuffer()
  const ctx = new (window.AudioContext || window.webkitAudioContext)()
  const audio = await ctx.decodeAudioData(buf)
  const ch = audio.getChannelData(0)
  const step = Math.max(Math.floor(ch.length / 2000), 1)
  const samples = []
  for (let i = 0; i < ch.length; i += step) samples.push(ch[i])
  waveData = { samples, duration: audio.duration }
  ctx.close()
}
function wavePos(e) {
  const r = waveEl.value.getBoundingClientRect()
  return Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1) * waveData.duration
}
function onSegStart(e) {
  const t = wavePos(e)
  for (let i = anns.value.length - 1; i >= 0; i--) {
    const a = anns.value[i]
    if (a.type === 'segment' && t >= a.start && t <= a.end) { selectedIdx.value = i; drawWave(); return }
  }
  segDrag = { t1: t, t2: t }
}
function onSegMove(e) { if (segDrag) { segDrag.t2 = wavePos(e); drawWave() } }
function onSegEnd() {
  if (!segDrag) return
  const [a, b] = [segDrag.t1, segDrag.t2]
  segDrag = null
  if (Math.abs(b - a) < 0.05) { drawWave(); return }
  anns.value.push({ type: 'segment', start: Math.min(a, b), end: Math.max(a, b),
                    label: currentLabel.value || '片段', text: '' })
  selectedIdx.value = anns.value.length - 1
  drawWave()
}
function drawWave() {
  const cv = waveEl.value
  if (!cv || !waveData) return
  cv.width = cv.clientWidth || 900
  cv.height = 200
  const ctx = cv.getContext('2d')
  const { samples, duration } = waveData
  ctx.fillStyle = '#0f1729'
  ctx.fillRect(0, 0, cv.width, cv.height)
  const t2x = (t) => (t / duration) * cv.width
  // 已标注段
  anns.value.forEach((a, i) => {
    if (a.type !== 'segment') return
    ctx.fillStyle = i === selectedIdx.value ? 'rgba(239,68,68,0.25)' : 'rgba(251,191,36,0.18)'
    ctx.fillRect(t2x(a.start), 0, t2x(a.end) - t2x(a.start), cv.height)
    ctx.fillStyle = '#fbbf24'
    ctx.font = '11px sans-serif'
    if (a.label) ctx.fillText(a.label, t2x(a.start) + 4, 14)
  })
  if (segDrag) {
    ctx.fillStyle = 'rgba(52,211,153,0.25)'
    ctx.fillRect(t2x(Math.min(segDrag.t1, segDrag.t2)), 0, Math.abs(t2x(segDrag.t2) - t2x(segDrag.t1)), cv.height)
  }
  // 波形
  ctx.strokeStyle = '#60a5fa'
  ctx.beginPath()
  const mid = cv.height / 2
  samples.forEach((v, i) => {
    const x = (i / samples.length) * cv.width
    ctx.moveTo(x, mid - v * mid * 0.9)
    ctx.lineTo(x, mid + v * mid * 0.9)
  })
  ctx.stroke()
  // 时间刻度
  ctx.fillStyle = '#64748b'
  ctx.font = '10px sans-serif'
  for (let t = 0; t <= duration; t += Math.max(Math.round(duration / 10), 1)) {
    ctx.fillText(`${t}s`, t2x(t) + 2, cv.height - 4)
  }
}

// ---------- 文本：划词 ----------
const textEl = ref(null)
const textContent = ref('')
const textRender = computed(() => {
  const text = textContent.value
  if (!text) return []
  const marks = anns.value.filter((a) => a.type === 'span')
    .sort((a, b) => a.start - b.start)
  const segs = []
  let pos = 0
  for (const m of marks) {
    if (m.start < pos || m.start >= text.length) continue
    if (m.start > pos) segs.push({ text: text.slice(pos, m.start) })
    segs.push({ text: text.slice(m.start, Math.min(m.end, text.length)), mark: true, idx: anns.value.indexOf(m) })
    pos = Math.min(m.end, text.length)
  }
  if (pos < text.length) segs.push({ text: text.slice(pos) })
  return segs
})
async function loadText() {
  const res = await fetch(fileUrl.value, { credentials: 'include' })
  const bytes = await res.arrayBuffer()
  let raw
  try { raw = new TextDecoder("utf-8", { fatal: true }).decode(bytes) }
  catch { raw = new TextDecoder("gb18030", { fatal: true }).decode(bytes) }
  if (raw.length > 200000) { ElMessage.error("文本超过20万字符，请先拆分再标注"); wbVisible.value = false; return }
  textContent.value = raw
}
function onTextSelect() {
  const sel = window.getSelection()
  if (!sel || sel.isCollapsed) return
  const node = textEl.value
  if (!node || !node.contains(sel.anchorNode)) return
  // 计算选区在全文中的偏移
  const range = sel.getRangeAt(0)
  const pre = range.cloneRange()
  pre.selectNodeContents(node)
  pre.setEnd(range.startContainer, range.startOffset)
  const start = pre.toString().length
  const text = sel.toString()
  if (!text.trim()) return
  anns.value.push({ type: 'span', start, end: start + text.length,
                    text, offset_unit: 'utf16', label: currentLabel.value || '实体' })
  selectedIdx.value = anns.value.length - 1
  sel.removeAllRanges()
}

// ---------- 雷达：BEV 点云框选 ----------
const bevEl = ref(null)
let bevData = null           // {points, xMin, xMax, yMin, yMax}

async function loadBev() {
  const res = await fetch(fileUrl.value, { credentials: 'include' })
  const text = await res.text()
  const pts = []
  const name = currentFile.value.name.toLowerCase()
  if (name.endsWith('.csv')) {
    const lines = text.split('\n')
    let header = null
    for (const line of lines) {
      const cols = line.split(',')
      if (header === null) {
        if (cols.every((c) => isNaN(parseFloat(c)))) { header = cols.map((c) => c.trim().toLowerCase()); continue }
        header = []
      }
      let x, y
      if (header.length) {
        const xi = header.findIndex((h) => ['x', 'range', '距离'].some((k) => h.includes(k)))
        const yi = header.findIndex((h) => ['y', 'azimuth', '方位'].some((k) => h.includes(k)))
        x = parseFloat(cols[xi >= 0 ? xi : 0]); y = parseFloat(cols[yi >= 0 ? yi : 1])
      } else { x = parseFloat(cols[0]); y = parseFloat(cols[1]) }
      if (!isNaN(x) && !isNaN(y)) pts.push([x, y])
    }
  } else if (name.endsWith('.ply')) {
    for (const line of text.split('\n')) {
      const cols = line.trim().split(/\s+/)
      if (cols.length >= 3 && !isNaN(parseFloat(cols[0]))) pts.push([parseFloat(cols[0]), parseFloat(cols[1])])
    }
  }
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1])
  bevData = {
    points: pts.length > 20000 ? pts.filter((_, i) => i % Math.ceil(pts.length / 20000) === 0) : pts,
    xMin: Math.min(...xs, 0), xMax: Math.max(...xs, 1),
    yMin: Math.min(...ys, 0), yMax: Math.max(...ys, 1),
  }
  // 已有 BEV 标注 → 画布坐标由 drawBev 换算；natW/natH 设为数据域
  nextTick(() => {
    const cv = bevEl.value
    if (cv) { cv.width = cv.clientWidth || 900; cv.height = 520 }
    natW = bevData.xMax - bevData.xMin
    natH = bevData.yMax - bevData.yMin
    cvEl.value = bevEl.value   // 拖框逻辑复用
    drawBev()
  })
}
function drawBev() {
  const cv = bevEl.value
  if (!cv || !bevData) return
  const ctx = cv.getContext('2d')
  ctx.fillStyle = '#0f1729'
  ctx.fillRect(0, 0, cv.width, cv.height)
  const x2c = (x) => ((x - bevData.xMin) / (bevData.xMax - bevData.xMin)) * cv.width
  const y2c = (y) => cv.height - ((y - bevData.yMin) / (bevData.yMax - bevData.yMin)) * cv.height
  ctx.fillStyle = 'rgba(96,165,250,0.6)'
  for (const [x, y] of bevData.points) ctx.fillRect(x2c(x), y2c(y), 2, 2)
  // 标注框
  anns.value.forEach((a, i) => {
    if (a.center) {
      ctx.fillStyle = i === selectedIdx.value ? '#ef4444' : '#fbbf24'
      ctx.beginPath(); ctx.arc(x2c(a.center[0]), y2c(a.center[1]), 5, 0, Math.PI * 2); ctx.fill(); return
    }
    if (a.type !== 'bev_box') return
    const x1 = x2c(a.range[0]), x2 = x2c(a.range[1])
    const y1 = y2c(a.azimuth[1]), y2 = y2c(a.azimuth[0])
    ctx.strokeStyle = i === selectedIdx.value ? '#ef4444' : '#fbbf24'
    ctx.lineWidth = 2
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
    drawTag(ctx, a.label, x1, y1)
  })
  if (drag) {
    ctx.strokeStyle = '#34d399'
    ctx.setLineDash([4, 4])
    ctx.strokeRect(drag.x1, drag.y1, drag.x2 - drag.x1, drag.y2 - drag.y1)
    ctx.setLineDash([])
  }
  // 坐标轴标注
  ctx.fillStyle = '#64748b'
  ctx.font = '10px sans-serif'
  ctx.fillText(`X: ${bevData.xMin.toFixed(0)} ~ ${bevData.xMax.toFixed(0)} m`, 8, cv.height - 6)
  ctx.fillText(`Y: ${bevData.yMin.toFixed(0)} ~ ${bevData.yMax.toFixed(0)} m`, 8, 14)
}

onMounted(async () => {
  await loadDatasets()
  const input=routeInput()
  if(input){
    datasetId.value=input.dataset_id
    await onDatasetChange()
    inputVersionId.value=input.version_id
    await loadFiles()
  }
})
</script>

<style scoped>
.hint { font-size:12px; color:#94a3b8; }
.wb-toolbar { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; flex-wrap: wrap; }
.video-workspace { display:grid; grid-template-columns:minmax(0,1fr) 320px; gap:18px; align-items:start; }
.video-picture { min-width:0; }
.video-results { border:1px solid #334155; border-radius:8px; padding:14px; }
.annotation-status { margin-bottom:12px; border:1px solid currentColor; }
.video-results h4 { margin:0 0 12px; display:flex; gap:8px; align-items:center; }
.current-annotation { padding-bottom:14px; margin-bottom:14px; border-bottom:1px solid #334155; }
.current-annotation dl { display:grid; grid-template-columns:58px 1fr; gap:8px; font-size:13px; }
.current-annotation dt { color:#94a3b8; }.current-annotation dd { margin:0; overflow-wrap:anywhere; }
.point-settings,.draft-actions { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-top:10px; font-size:12px; }
.annotation-result-list { max-height:42vh; overflow:auto; }
.annotation-result { display:flex; flex-direction:column; gap:5px; width:100%; text-align:left; background:transparent; color:inherit; border:1px solid #334155; border-radius:5px; padding:10px; margin-top:8px; cursor:pointer; }
div.annotation-result { cursor:default; }
.annotation-result strong { display:flex; justify-content:space-between; }.annotation-result span,.annotation-result small { color:#94a3b8; font-size:12px; }
.annotation-actions { display:flex; gap:8px; margin-top:4px; }.annotation-actions .el-button + .el-button { margin-left:0; }
.annotation-result.editing,.annotation-result.previewing { border-color:#409eff; }.draft-error { color:#f87171; font-size:12px; }
@media(max-width:1050px){.video-workspace{grid-template-columns:minmax(0,1fr)}.annotation-result-list{max-height:200px}}
.stage { position: relative; display: inline-block; max-width: 100%; }
.stage img { max-width: 100%; max-height: 62vh; display: block; user-select: none; }
.draw-layer { position: absolute; left: 0; top: 0; cursor: crosshair; }
.wave-canvas { width: 100%; height: 200px; border: 1px solid #24304d; border-radius: 8px; cursor: crosshair; }
.bev-canvas { width: 100%; height: 520px; border: 1px solid #24304d; border-radius: 8px; cursor: crosshair; }
.text-stage {
  background: #0f1729; border: 1px solid #24304d; border-radius: 8px;
  padding: 14px; max-height: 60vh; overflow: auto; white-space: pre-wrap;
  font-size: 0.85rem; color: #cbd5e1; line-height: 1.8; cursor: text;
}
.hl { background: rgba(251, 191, 36, 0.35); color: #fbbf24; border-radius: 2px; padding: 0 1px; cursor: pointer; }
.hl.sel { background: rgba(239, 68, 68, 0.45); color: #ef4444; }
.ann-list { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 6px; }
</style>
