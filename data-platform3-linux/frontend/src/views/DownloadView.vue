<template>
  <div class="page">
    <div class="page-header">
      <h2><span>📦</span> 数据下载</h2>
      <p>场景、标签、项目、密级任意填写一个即可查询；数量不足时可申请 AIGC 生成补足。</p>
    </div>

    <div class="filter-bar download-filter">
      <el-form inline class="query-form">
        <el-form-item label="模态">
          <el-select v-model="query.data_type" placeholder="选择模态" style="width: 120px" @change="onDataTypeChange">
            <el-option v-for="t in typeOptions" :key="t.value" :label="t.label" :value="t.value" />
          </el-select>
        </el-form-item>
        <el-form-item label="场景">
          <el-select v-model="query.scene" clearable filterable allow-create default-first-option
                     placeholder="全部场景" style="width: 160px">
            <el-option v-for="s in sceneOptions" :key="s" :label="s" :value="s" />
          </el-select>
        </el-form-item>
        <el-form-item label="标签">
          <el-select v-model="query.labels" multiple filterable placeholder="如：卡车" style="width: 220px">
            <el-option v-for="l in labelOptions" :key="l" :label="l" :value="l" />
          </el-select>
        </el-form-item>
        <el-form-item label="项目">
          <el-select v-model="query.project" clearable filterable placeholder="全部项目" style="width: 160px">
            <el-option v-for="p in projectOptions" :key="p" :label="p" :value="p" />
          </el-select>
        </el-form-item>
        <el-form-item label="密级">
          <el-select v-model="query.secret_level" clearable placeholder="全部" style="width: 120px">
            <el-option v-for="s in secretOptions" :key="s" :label="s" :value="s" />
          </el-select>
        </el-form-item>
        <el-form-item label="需求数量">
          <el-input-number v-model="query.required_count" :min="1" :max="100000" style="width: 140px" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :loading="loading" @click="runQuery">查询</el-button>
        </el-form-item>
      </el-form>
    </div>

    <div class="filter-bar request-history">
      <span>下载请求</span>
      <el-select v-model="requestId" placeholder="选择已保存的查询" filterable style="width: 360px" @change="loadRequest(true)">
        <el-option v-for="r in requests" :key="r.id" :value="r.id" :label="`#${r.id} · ${r.query.scene || r.query.labels.join('、') || '全部条件'} · ${r.owner_name}`" />
      </el-select>
      <el-checkbox v-if="auth.canViewUsers" v-model="allUsers" @change="loadRequests">查看所有用户</el-checkbox>
      <el-button :disabled="!requestId" @click="loadRequest(false)">刷新任务与版本</el-button>
    </div>

    <template v-if="queried">
      <el-alert v-if="shortage > 0" type="warning" :closable="false" class="shortage-alert"
                :title="`命中 ${result.matched_count} 条，距需求数量还差 ${shortage} 条`"
                description="可申请 AIGC 生成补足（生成数据将标记为 AIGC 来源）">
        <template #default>
          <div v-if="activeQuery.data_type === 'image'" class="aigc-row">
            <el-input-number v-model="aigcCount" :min="1" :max="Math.min(shortage, 1000)" style="width: 140px" />
            <el-button type="warning" plain :loading="aigcLoading" :disabled="!auth.canOperate || currentRequest?.created_by !== auth.user?.id" @click="runAigc">AIGC 生成补足</el-button>
          </div>
          <p v-if="activeQuery.data_type === 'image'" class="hint">带入本次查询命中的候选图像；确认素材、训练文本、模型和数量后才开始生成。一次最多补充 1000 张。</p>
          <p v-else class="hint">少样本补充目前支持图像数据。</p>
        </template>
      </el-alert>
      <el-alert v-else type="success" :closable="false" class="shortage-alert"
                :title="`命中 ${result.matched_count} 条文件，满足需求数量`" />

      <div class="table-card result-card">
        <el-table :data="result.files" border size="small" max-height="420">
          <el-table-column prop="file_name" label="文件名" min-width="240" show-overflow-tooltip />
          <el-table-column prop="dataset_name" label="所属数据集" min-width="140" show-overflow-tooltip />
          <el-table-column prop="data_type" label="模态" width="90" />
          <el-table-column prop="source_type" label="来源" width="90" />
          <el-table-column prop="scene" label="场景" width="120" />
          <el-table-column prop="secret_level" label="密级" width="90" />
          <el-table-column label="命中标签" min-width="160">
            <template #default="{ row }">
              <el-tag v-for="l in row.matched_labels" :key="l" size="small" style="margin-right: 4px">{{ l }}</el-tag>
            </template>
          </el-table-column>
        </el-table>

        <div class="footer-row">
          <span class="hint">打包内容：数据文件 + 对应标签文件 + _lineage.json 血缘清单</span>
          <el-button type="primary" :disabled="!result.files.length" :loading="packing" @click="runPack">
            打包下载（{{ result.files.length }} 条）
          </el-button>
        </div>
      </div>
    </template>

    <div v-else class="table-card result-empty">
      <el-empty description="设置条件后点击查询" :image-size="80" />
    </div>

    <section v-if="currentRequest" class="generated-section">
      <div class="generated-heading"><h3>本次 AIGC 补充</h3><span class="hint">下载请求 #{{ currentRequest.id }} · 所属用户：{{ currentRequest.owner_name }}</span></div>
      <p class="hint">任务与本次查询分别关联。生成数量和符合标签条件的数量分别统计；预处理完成后可选择新的版本下载。</p>
      <el-empty v-if="!currentRequest.jobs.length" description="尚未提交补充任务，点击上方 AIGC 生成补足后确认素材" :image-size="65" />
      <article v-for="job in currentRequest.jobs" :key="job.id" class="table-card generated-job">
        <div class="generated-heading"><strong>{{ job.name }} · #{{ job.id }}</strong><el-tag :type="job.status==='failed'?'danger':job.status==='completed'?'success':'info'">{{ statusText(job.status) }}</el-tag></div>
        <p class="hint">{{ job.engine_name }} · 计划生成 {{ job.groups.reduce((n,g)=>n+g.count,0) }} 张 · AIGC 来源</p>
        <p v-if="job.error" class="error-message">{{ job.error }}</p>
        <div class="result-actions">
          <el-button @click="router.push({path:'/fewshot',query:{job_id:job.id,download_request_id:currentRequest.id}})">查看生成任务</el-button>
          <template v-if="job.dataset_id && job.versions.length">
            <el-select v-model="selectedVersions[job.id]" aria-label="下载结果版本" style="width: 250px">
              <el-option v-for="v in job.versions" :key="v.id" :value="v.id" :disabled="!v.ready" :label="`V${v.version_no} · ${v.stage} · ${v.data_count} 个文件${v.status==='published'?' · 已发布':''}${!v.ready?' · 处理中或未成功完成':''}`" />
            </el-select>
            <el-tag type="warning" effect="plain">待人工核验</el-tag>
            <el-button type="primary" :disabled="job.status!=='completed' || !job.versions.find(v=>v.id===selectedVersions[job.id])?.ready" @click="downloadGenerated(job)">下载所选版本</el-button>
          </template>
        </div>
        <div v-if="job.dataset_id && job.versions.length" class="result-actions preprocess-links">
          <el-button v-for="entry in processingEntries" :key="entry.path" size="small" :disabled="!job.versions.find(v=>v.id===selectedVersions[job.id])?.ready" @click="openProcess(job,entry.path)">{{ entry.label }}</el-button>
        </div>
      </article>
    </section>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth'
import { ElMessage } from 'element-plus'
import api from '../api'

const auth=useAuthStore(),router=useRouter(),route=useRoute()
const requestId=ref(null),requests=ref([]),currentRequest=ref(null),allUsers=ref(false),selectedVersions=ref({})
const processingEntries=[{path:'manual-label',label:'人工修正'},{path:'auto-label',label:'自动标注'},
  {path:'clean',label:'清洗'},{path:'transform',label:'转换'},{path:'sample',label:'采样'},
  {path:'integrate',label:'集成'},{path:'workflow',label:'工作流'},{path:'quality',label:'质量评估'}]
let timer,stopped=false,requestSerial=0
const storageKey=()=>`dp_download_request_${auth.user?.id}`
const typeOptions = [
  { label: '语音', value: 'voice' }, { label: '雷达', value: 'radar' },
  { label: '图像', value: 'image' }, { label: '视频', value: 'video' },
  { label: '文本', value: 'text' },
]
const secretOptions = ['公开', '内部', '秘密', '机密']
const projectOptions = ref([])
const sceneOptions = ref([])

const query = ref({
  data_type: 'image', scene: '', labels: [], project: null,
  secret_level: null, required_count: 100,
})
const labelOptions = ref([])
const result = ref({ files: [], matched_count: 0 })
const queried = ref(false)
const loading = ref(false)
const packing = ref(false)
const aigcLoading = ref(false)
const aigcCount = ref(0)

const activeQuery=computed(()=>currentRequest.value?.query||query.value)
const shortage = computed(() => Math.max(0, activeQuery.value.required_count - (result.value.matched_count || 0)))

async function loadProjects() {
  try {
    const r = await api.get('/projects')
    projectOptions.value = (r.items || []).map(p => p.name)
  } catch { projectOptions.value = [] }
}

async function loadScenes() {
  try {
    const r = await api.get('/download/scenes', { params: { data_type: query.value.data_type } })
    sceneOptions.value = r.items || []
  } catch { sceneOptions.value = [] }
}

function onDataTypeChange() {
  query.value.scene = ''
  loadLabels()
  loadScenes()
}

async function loadLabels() {
  query.value.labels = []
  try {
    const r = await api.get('/labels', { params: { data_type: query.value.data_type } })
    labelOptions.value = r.labels || []
  } catch { labelOptions.value = [] }
}

async function runQuery() {
  loading.value = true
  try {
    const payload = {
      ...query.value,
      scene: query.value.scene || '',
      labels: query.value.labels || [],
      project: query.value.project || '',
      secret_level: query.value.secret_level || '',
    }
    const r = await api.post('/download/query', payload)
    result.value = r
    queried.value = true
    aigcCount.value = Math.min(Math.max(0,payload.required_count-r.matched_count),1000)
    requestId.value=r.download_request_id
    await loadRequest(false)
    await loadRequests()
  } catch (e) {
    ElMessage.error(e?.response?.data?.detail || '查询失败')
  } finally { loading.value = false }
}

async function runAigc() {
  if(!requestId.value)return
  aigcLoading.value=true
  try {
    await api.post('/aigc/generate',{download_request_id:requestId.value,count:aigcCount.value})
    router.push({path:'/fewshot',query:{download_request_id:requestId.value}})
  } finally {aigcLoading.value=false}
}

async function loadRequests(){
  requests.value=(await api.get('/download/requests',{params:{all_users:allUsers.value}})).items
}
async function loadRequest(restore=false){
  if(!requestId.value)return
  const serial=++requestSerial,id=requestId.value
  try{
    const r=await api.get(`/download/requests/${id}`)
    if(stopped||serial!==requestSerial)return
    currentRequest.value=r
    if(r.created_by!==auth.user?.id&&auth.canViewUsers&&!allUsers.value){allUsers.value=true;await loadRequests()}
    if(restore){query.value={...r.query};aigcCount.value=Math.min(r.generation_count||1000,Math.max(1,r.query.required_count-r.current_matched_count));await loadScenes()}
    result.value={files:r.files,matched_count:r.current_matched_count};queried.value=true
    for(const j of r.jobs)if(!j.versions.some(v=>v.id===selectedVersions.value[j.id]&&v.ready))selectedVersions.value[j.id]=(j.versions.find(v=>v.id===j.current_version_id&&v.ready)||j.versions.find(v=>v.ready))?.id
    localStorage.setItem(storageKey(),String(id))
  }catch{if(serial===requestSerial){currentRequest.value=null;requestId.value=null;queried.value=false;localStorage.removeItem(storageKey())}}
}
function statusText(s){return ({draft:'待确认提交',queued:'排队中',training:'训练中',generating:'生成中',importing:'入库中',labeling:'自动标注中',completed:'可下载 · 待人工核验',failed:'执行失败'})[s]||s}
function openProcess(job,path){router.push({path:`/${path}`,query:{dataset_id:job.dataset_id,version_id:selectedVersions.value[job.id],download_request_id:requestId.value}})}
async function downloadGenerated(job){
  const r=await api.get(`/datasets/${job.dataset_id}/versions/${selectedVersions.value[job.id]}/download`,{responseType:'blob'})
  saveBlob(r,`generated_job_${job.id}_version_${selectedVersions.value[job.id]}.zip`)
}
function saveBlob(blob,name){const url=URL.createObjectURL(blob instanceof Blob?blob:new Blob([blob]));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}

async function runPack() {
  packing.value = true
  try {
    const r = await api.post('/download/pack', { file_ids: result.value.files.map(f => f.file_id) }, { responseType: 'blob' })
    const url = URL.createObjectURL(new Blob([r]))
    const a = document.createElement('a')
    a.href = url
    a.download = `download_${Date.now()}.zip`
    a.click()
    URL.revokeObjectURL(url)
  } catch (e) {
    ElMessage.error(e?.response?.data?.detail || '打包失败')
  } finally { packing.value = false }
}

onMounted(async()=>{
  await Promise.all([loadProjects(),loadScenes(),loadLabels(),loadRequests()])
  requestId.value=Number(route.query.download_request_id||localStorage.getItem(storageKey()))||null
  if(requestId.value)await loadRequest(true)
  timer=setInterval(()=>{if(currentRequest.value?.jobs.some(j=>['queued','training','generating','importing','labeling'].includes(j.status)))loadRequest(false)},5000)
})
onUnmounted(()=>{stopped=true;clearInterval(timer)})

</script>

<style scoped>
.request-history,.generated-heading,.result-actions{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.generated-section{margin-top:24px}.generated-heading{justify-content:space-between}.generated-job{margin-top:12px}.result-actions{margin-top:12px}.error-message{white-space:pre-wrap;color:var(--dp-danger);overflow-wrap:anywhere}.preprocess-links{padding-top:12px;border-top:1px solid var(--dp-border)}
.download-filter { padding-bottom: 2px; }
.query-form { width: 100%; }
.query-form :deep(.el-form-item) { margin-right: 18px; margin-bottom: 12px; }
.shortage-alert { margin-bottom: 14px; }
.aigc-row { margin-top: 8px; display: flex; gap: 8px; align-items: center; }
.result-card { padding: 0; overflow: hidden; }
.result-card :deep(.el-table) { --el-table-border-color: var(--dp-border); }
.footer-row {
  padding: 12px 14px;
  border-top: 1px solid var(--dp-border);
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.hint { color: var(--dp-text-secondary); font-size: 12px; }
.result-empty { min-height: 360px; display: grid; place-items: center; }
.result-empty :deep(.el-empty__description p) { color: var(--dp-text-secondary); }
</style>