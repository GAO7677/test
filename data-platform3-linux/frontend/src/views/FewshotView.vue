<template>
  <div class="page fewshot-page">
    <div class="page-header"><h2>少样本数据生成</h2><p>选择训练方法和基础模型；生成图片自动打框，再由你确认标注。</p></div>
    <el-alert v-if="capability && !capability.ready" :title="capability.error" type="warning" :closable="false" />
    <el-alert v-if="downloadRequest" type="info" :closable="false" class="download-context" :title="`关联下载请求 #${downloadRequest.id} · ${downloadRequest.candidates.length} 张候选图像`">
      <p v-if="downloadRequest.created_by===auth.user?.id">候选图像已经带入素材组。请确认、增删素材和分组，修改训练文本、方法和生成数量后再提交。</p><p v-else>当前查看其他用户的下载请求。管理员可查看关联任务；新增补充请使用自己的下载请求。</p>
      <el-button size="small" @click="router.push({path:'/download',query:{download_request_id:downloadRequest.id}})">返回对应的数据下载</el-button>
    </el-alert>
    <div class="workspace">
      <section class="table-card composer">
        <h3>素材与生成设置</h3>
        <el-form label-position="top" @submit.prevent="create">
          <div class="field-pair">
            <el-form-item label="训练方法"><el-select v-model="options.method" aria-label="训练方法" :disabled="busy" @change="changeMethod"><el-option label="BeyondObjects" value="beyondobjects" /><el-option label="ChimeraLoRA" value="chimeralora" /></el-select></el-form-item>
            <el-form-item label="基础模型"><el-select v-model="options.model" aria-label="基础模型" :disabled="busy" @change="changeModel"><el-option v-for="engine in modelChoices" :key="engine.id" :label="engine.model==='sd21'?'SD2.1':'FLUX.2 Klein Base 4B'" :value="engine.model" /></el-select></el-form-item>
          </div>
          <p class="hint engine-note">{{ selectedEngine?.note }}</p>
          <el-alert v-if="selectedEngine && !selectedEngine.ready" :title="selectedEngine.error" type="warning" :closable="false" />
          <el-form-item label="任务名称"><el-input v-model="form.name" maxlength="80" placeholder="例如：机场飞机图像扩充" /></el-form-item>
          <div class="field-pair">
            <el-form-item label="项目"><el-input v-model="form.project" maxlength="100" /></el-form-item>
            <el-form-item label="场景"><el-input v-model="form.scene" maxlength="100" placeholder="例如：机场俯视" /></el-form-item>
          </div>
          <el-form-item label="密级"><el-select v-model="form.secret_level" style="width:100%"><el-option v-for="level in levels" :key="level" :label="level" :value="level" /></el-select></el-form-item>
          <div v-for="(group, i) in groups" :key="group.key" class="material-group">
            <div class="group-heading"><strong>素材组 {{ i + 1 }}</strong><el-button v-if="groups.length > 1" text type="danger" :disabled="busy" @click="groups.splice(i,1)">移除</el-button></div>
            <el-input v-model="group.name" aria-label="分组名称" placeholder="分组名称，例如：中型客机" maxlength="80" />
            <label class="file-picker">选择图片<input type="file" accept=".png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff" multiple :disabled="busy" @change="chooseFiles(group,$event)" /></label>
            <el-button :disabled="busy || !auth.canOperate" @click="openMaterials(group)">从本地素材库选择</el-button>
            <el-button v-if="downloadRequest" :disabled="busy" @click="openCandidates(group)">调整查询候选素材</el-button>
            <span class="hint">{{ group.files.length + group.localImages.length + group.downloadFiles.length }} 张素材</span>
            <p v-if="group.files.length" class="file-names">{{ group.files.map(f => f.name).join('、') }}</p>
            <div v-if="group.localImages.length" class="local-selected"><el-tag v-for="item in group.localImages" :key="materialKey(item)" closable :disable-transitions="true" @close="removeMaterial(group,item)">{{ item.name }}</el-tag></div>
            <div v-if="group.downloadFiles.length" class="local-selected"><el-tag v-for="item in group.downloadFiles" :key="item.file_id" closable :disable-transitions="true" @close="removeCandidate(group,item)">{{ item.name }}</el-tag></div>
            <label :for="`caption-${group.key}`" class="caption-label">训练文本提示词（caption）</label>
            <el-input :id="`caption-${group.key}`" v-model="group.caption" type="textarea" :rows="3" maxlength="2000" aria-label="训练文本提示词" placeholder="填写适用于组内所有图片的描述：主体、视角、场景。建议使用英文。" />
            <div class="count-row"><span>本组生成数量</span><el-input-number v-model="group.count" :min="1" :max="1000" size="small" /></div>
          </div>
          <el-button :disabled="groups.length >= 20 || busy" @click="addGroup">＋ 添加素材组</el-button>
          <p class="hint">最多 100 张素材，单图不超过 20 MB，总计不超过 500 MB。各组共同训练，按各组训练文本提示词生成。</p>
          <p v-if="options.method==='beyondobjects'&&options.model==='sd21'" class="hint">SD2.1 使用完整训练轮次，当前素材将执行约 {{ effectiveSteps }} 步。</p>
          <el-collapse class="advanced"><el-collapse-item title="训练与采样参数" name="params"><div class="field-pair">
            <el-form-item label="训练步数"><el-input-number v-model="options.steps" :min="2" :max="4000" :step="100" /></el-form-item>
            <el-form-item label="LoRA rank"><el-select v-model="options.rank"><el-option v-for="n in [4,8,16]" :key="n" :value="n" :label="String(n)" /></el-select></el-form-item>
            <el-form-item label="图像边长"><el-select v-model="options.resolution"><el-option v-for="n in [512,768,1024]" :key="n" :value="n" :label="`${n} px`" /></el-select></el-form-item>
            <el-form-item label="采样步数"><el-input-number v-model="options.sampling_steps" :min="1" :max="100" /></el-form-item>
            <el-form-item label="随机种子"><el-input-number v-model="options.seed" :min="0" :max="2147480000" /></el-form-item>
          </div></el-collapse-item></el-collapse>
          <div class="submit-row"><span>预计生成 <strong>{{ totalImages }}</strong> 张</span><el-button type="primary" :loading="busy" :disabled="!auth.canOperate || !selectedEngine?.ready || (downloadRequest && downloadRequest.created_by!==auth.user?.id)" @click="create">开始训练与生成</el-button></div>
          <p v-if="busy" class="hint">{{ uploadProgress < 100 ? `上传中 ${uploadProgress}%` : '正在校验素材并提交任务…' }}</p>
        </el-form>
      </section>
      <section class="task-column">
        <div class="table-card task-list">
          <div class="section-heading"><h3>生成任务</h3><el-checkbox v-if="auth.canViewUsers" v-model="allUsers" @change="page=1;refresh()">查看所有用户</el-checkbox><el-button text @click="refresh">刷新</el-button></div>
          <el-empty v-if="!jobs.length" description="上传一组图像，开始第一次数据扩充" :image-size="80" />
          <button v-for="job in jobs" :key="job.id" class="job-row" :class="{selected:selected?.id===job.id}" @click="select(job.id)">
            <span><b>{{ job.name }}</b><small>#{{ job.id }} · {{ job.groups.length }} 组素材 · {{ job.engine_name }}{{ job.reuse_job_id ? ' · 复用权重' : '' }}{{ job.smoke ? ' · 流程验收' : '' }} · 用户 #{{ job.created_by }}</small></span>
            <el-tag :type="job.status==='failed'?'danger':job.status==='completed'?'success':'info'" size="small">{{ statusText(job.status) }}</el-tag>
          </button>
          <div class="pagination"><el-button size="small" :disabled="page===1" @click="changePage(-1)">上一页</el-button><span>第 {{ page }} 页</span><el-button size="small" :disabled="jobs.length<30" @click="changePage(1)">下一页</el-button></div>
        </div>
        <div v-if="selected" class="table-card detail">
          <div class="section-heading"><h3>{{ selected.name }}</h3><el-tag size="small">{{ selected.secret_level }}</el-tag></div>
          <el-button v-if="selected.download_request_id" size="small" @click="router.push({path:'/download',query:{download_request_id:selected.download_request_id}})">返回对应的数据下载</el-button>
          <p class="hint">{{ selected.engine_name }}{{ selected.engine==='chimera_sd21'?' · 共享 A + 多 B 头':'' }} · {{ selected.effective_steps }} 训练步</p>
          <p v-if="selected.status==='queued'&&selected.waiting?.free_mb!=null" class="hint">等待显存：可用 {{ (selected.waiting.free_mb/1024).toFixed(1) }} GB，需要至少 {{ (selected.waiting.required_mb/1024).toFixed(1) }} GB；正在连续检查空闲余量。</p>
          <div class="pipeline" aria-label="任务处理流程"><div v-for="(step,i) in stages" :key="step.key" :class="{done:stageIndex>i,current:stageIndex===i}"><span>{{ i+1 }}</span><b>{{ step.label }}</b></div></div>
          <p>{{ statusText(selected.status) }}<span v-if="selected.progress.loss!=null"> · loss {{ Number(selected.progress.loss).toFixed(4) }}</span></p>
          <el-progress v-if="selected.status==='training'" :percentage="trainPercent" />
          <p v-if="selected.status==='training'" class="hint">训练 {{ selected.progress.step||0 }} / {{ selected.progress.total_steps||selected.options.steps }} 步</p>
          <p v-if="selected.status==='generating'" class="hint">已生成 {{ selected.progress.images||0 }} / {{ selected.progress.total_images||selected.groups.reduce((n,g)=>n+g.count,0) }} 张</p>
          <el-alert v-if="selected.error" :title="selected.error" type="error" :closable="false" />
          <el-alert v-if="selected.status==='completed'" title="自动标注完成，等待人工修正" :description="`全部生成图已入库。${selected.empty_labels} 张未检测到目标，请人工检查。自动框不代表已审核。`" type="success" :closable="false" />
          <div class="actions">
            <el-button v-if="['draft','failed','queued'].includes(selected.status)&&!selected.smoke" type="primary" :loading="acting" :disabled="!auth.canOperate" @click="submit(selected.id)">{{ selected.status==='failed'?'重试未完成阶段':selected.status==='queued'?'唤醒任务队列':'提交任务' }}</el-button>
            <el-button v-if="selected.dataset_id" type="primary" @click="router.push({path:'/manual-label',query:{dataset_id:selected.dataset_id,version_id:selected.current_version_id||selected.version_id}})">进入人工修正</el-button>
            <el-button v-if="selected.status==='completed'&&!selected.smoke" :loading="acting" :disabled="!auth.canOperate" @click="regenerate">复用权重再生成</el-button>
          </div>
          <div v-if="selected.images?.length" class="result-grid"><figure v-for="name in selected.images" :key="name"><img :src="`/api/fewshot/jobs/${selected.id}/images/${name}`" :alt="`生成图像 ${name}`" loading="lazy" /><figcaption>{{ name }}</figcaption></figure></div>
          <el-collapse><el-collapse-item title="查看运行日志" name="log"><pre>{{ selected.log||'任务尚未开始执行。' }}</pre></el-collapse-item></el-collapse>
        </div>
      </section>
    </div>
    <el-dialog v-model="materialOpen" title="选择本地训练素材" width="900px" top="5vh" class="material-dialog">
      <p class="hint">选中的图片加入当前素材组，共用该组描述。提交后保存素材副本；原始数据不变。</p>
      <div class="material-toolbar">
        <el-select v-model="materialDataset" aria-label="本地数据集" @change="changeMaterialDataset"><el-option v-for="ds in materialCatalog" :key="ds.id" :value="ds.id" :label="ds.name" :disabled="!ds.available" /></el-select>
        <el-select-v2 v-model="materialCategory" aria-label="素材类别" filterable :options="materialCategoryOptions" placeholder="全部类别" @change="loadMaterials(1)" />
        <el-input v-model="materialQuery" placeholder="搜索文件名或描述" aria-label="素材文件名或描述" clearable @keyup.enter="loadMaterials(1)" @clear="loadMaterials(1)" />
        <el-button @click="loadMaterials(1)">搜索</el-button>
      </div>
      <div v-loading="materialLoading" class="material-results">
        <section v-for="section in materialSections" :key="section.name" class="material-section">
          <h4>{{ section.name }}</h4>
          <div class="material-grid">
            <article v-for="item in section.items" :key="item.name" class="material-tile" :class="{picked:isPicked(item)}">
              <button type="button" class="material-select" :aria-label="`选择图片 ${item.name}`" :aria-pressed="isPicked(item)" @click="toggleMaterial(item)">
                <img :src="`/api/fewshot/materials/${materialDataset}/thumbnail?name=${encodeURIComponent(item.name)}`" :alt="item.name" loading="lazy" />
                <span>{{ isPicked(item) ? '✓ 已选择' : '选择图片' }}</span>
              </button>
              <div class="material-description">
                <b v-if="item.categories?.length" class="material-label">{{ item.categories.join('、') }}</b>
                <p v-if="item.caption" class="material-caption">{{ item.caption }}</p>
                <p v-else-if="!item.categories?.length" class="hint">没有类别或描述</p>
                <span class="material-filename">{{ item.name }}</span>
                <small>{{ (item.size/1024/1024).toFixed(2) }} MB</small>
              </div>
            </article>
          </div>
        </section>
      </div>
      <el-empty v-if="!materialLoading && !materialItems.length" description="没有匹配的图片" />
      <div class="pagination"><el-button :disabled="materialPage===1 || materialLoading" @click="loadMaterials(materialPage-1)">上一页</el-button><span>{{ materialPage }} / {{ Math.max(1,Math.ceil(materialTotal/24)) }} · 共 {{ materialTotal }} 张</span><el-button :disabled="materialPage*24>=materialTotal || materialLoading" @click="loadMaterials(materialPage+1)">下一页</el-button></div>
      <template #footer><span class="hint">本组选中 {{ materialPicked.length }} 张 </span><el-button @click="materialOpen=false">取消</el-button><el-button type="primary" @click="confirmMaterials">加入素材组</el-button></template>
    </el-dialog>
    <el-dialog v-model="candidateOpen" title="调整查询候选素材" width="850px" class="material-dialog">
      <p class="hint">当前组 {{ candidatePicked.length }} 张；已在其他组中的素材不可重复选择。可移除后重新分组。</p>
      <div class="material-results"><div class="material-grid">
        <button v-for="item in downloadRequest?.candidates || []" :key="item.file_id" class="material-tile candidate-tile" :class="{picked:candidatePicked.some(x=>x.file_id===item.file_id)}" :disabled="candidateUsedElsewhere(item)" @click="toggleCandidate(item)">
          <img :src="`/api/download/requests/${downloadRequest.id}/candidates/${item.file_id}/thumbnail`" loading="lazy" />
          <span class="material-description"><b>{{ item.name }}</b><p v-if="item.caption" class="material-caption">{{ item.caption }}</p><small>{{ candidateUsedElsewhere(item)?'已在其他组中':'点击选择或移除' }}</small></span>
        </button>
      </div></div>
      <template #footer><el-button @click="candidateOpen=false">取消</el-button><el-button type="primary" @click="confirmCandidates">使用所选素材</el-button></template>
    </el-dialog>
  </div>
</template>
<script setup>
import { computed,onMounted,onUnmounted,reactive,ref } from 'vue'
import { useRouter,useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useAuthStore } from '../stores/auth'
import api from '../api'
const auth=useAuthStore(),router=useRouter(),route=useRoute()
const form=reactive({name:'',project:'',scene:'',secret_level:'内部'})
const options=reactive({method:'beyondobjects',model:'flux2_klein_base_4b',steps:400,rank:4,seed:0,resolution:512,sampling_steps:30})
const levels=['公开','受控','内部','秘密','机密','绝密']
const groups=ref([]),jobs=ref([]),selected=ref(null),capability=ref(null)
const modelChoices=computed(()=>capability.value?.engines?.filter(e=>e.method===options.method)||[])
const selectedEngine=computed(()=>modelChoices.value.find(e=>e.model===options.model))
const effectiveSteps=computed(()=>{
  const perEpoch=Math.max(1,Math.ceil(groups.value.reduce((n,g)=>n+g.files.length+g.localImages.length+g.downloadFiles.length,0)/4))
  return Math.ceil(options.steps/perEpoch)*perEpoch
})
function changeMethod(){if(!modelChoices.value.some(e=>e.model===options.model))options.model=modelChoices.value[0]?.model||'sd21';requestKey=null}
function changeModel(){requestKey=null}
const busy=ref(false),acting=ref(false),uploadProgress=ref(0),page=ref(1)
const materialOpen=ref(false),materialCatalog=ref([]),materialDataset=ref(''),materialQuery=ref(''),materialItems=ref([])
const materialPage=ref(1),materialTotal=ref(0),materialLoading=ref(false),materialPicked=ref([])
const materialCategory=ref(''),materialCategories=ref([])
const materialCategoryOptions=computed(()=>[{label:'全部类别',value:''},...materialCategories.value.map(cat=>({label:`${cat.name} · ${cat.count} 张`,value:cat.value||cat.name}))])
const materialSections=computed(()=>{
  const sections=new Map()
  for(const item of materialItems.value){
    const name=(materialCategory.value && materialCategory.value!=='__uncategorized__' ? materialCategory.value : item.primary_category) || '未分类'
    if(!sections.has(name))sections.set(name,[])
    sections.get(name).push(item)
  }
  return Array.from(sections,([name,items])=>({name,items}))
})
let materialGroup=null,materialRequest=0
function materialKey(item){return `${item.dataset}/${item.name}`}
function isPicked(item){return materialPicked.value.some(x=>materialKey(x)===materialKey({...item,dataset:materialDataset.value}))}
async function openMaterials(group){
  materialGroup=group;materialPicked.value=group.localImages.map(x=>({...x}));materialQuery.value='';materialCategory.value=''
  const result=await api.get('/fewshot/materials');materialCatalog.value=result.items
  materialDataset.value ||= result.items.find(x=>x.available)?.id || ''
  materialOpen.value=true
  if(materialDataset.value)await loadMaterials(1)
}
async function changeMaterialDataset(){materialCategory.value='';materialCategories.value=[];materialQuery.value='';await loadMaterials(1)}
async function loadMaterials(p){
  const request=++materialRequest;materialLoading.value=true;materialItems.value=[]
  try{const result=await api.get(`/fewshot/materials/${materialDataset.value}`,{params:{page:p,q:materialQuery.value,category:materialCategory.value}})
    if(request===materialRequest){materialItems.value=result.items;materialTotal.value=result.total;materialPage.value=p;materialCategories.value=result.categories||[]}
  }finally{if(request===materialRequest)materialLoading.value=false}
}
function toggleMaterial(item){
  const ref={...item,dataset:materialDataset.value},key=materialKey(ref)
  if(isPicked(item)){materialPicked.value=materialPicked.value.filter(x=>materialKey(x)!==key);return}
  const other=groups.value.flatMap(g=>g===materialGroup?[...g.files,...g.downloadFiles]:[...g.files,...g.localImages,...g.downloadFiles])
  if(item.size>20*1024**2 || other.length+materialPicked.value.length>=100 || [...other,...materialPicked.value,item].reduce((n,f)=>n+f.size,0)>500*1024**2)return ElMessage.warning('最多 100 张素材，单图 20 MB，总计 500 MB')
  materialPicked.value.push(ref)
}
function confirmMaterials(){materialGroup.localImages=[...materialPicked.value];requestKey=null;materialOpen.value=false}
function removeMaterial(group,item){if(busy.value)return;group.localImages=group.localImages.filter(x=>materialKey(x)!==materialKey(item));requestKey=null}
const downloadRequest=ref(null),allUsers=ref(false),candidateOpen=ref(false),candidatePicked=ref([])
let candidateGroup
function openCandidates(group){candidateGroup=group;candidatePicked.value=[...group.downloadFiles];candidateOpen.value=true}
function candidateUsedElsewhere(item){return groups.value.some(g=>g!==candidateGroup&&g.downloadFiles.some(x=>x.file_id===item.file_id))}
function toggleCandidate(item){
  const at=candidatePicked.value.findIndex(x=>x.file_id===item.file_id)
  if(at>=0){candidatePicked.value.splice(at,1);return}
  const other=groups.value.flatMap(g=>g===candidateGroup?[...g.files,...g.localImages]:[...g.files,...g.localImages,...g.downloadFiles])
  if(other.length+candidatePicked.value.length>=100||[...other,...candidatePicked.value,item].reduce((n,x)=>n+x.size,0)>500*1024**2)return ElMessage.warning('最多 100 张素材，总计不超过 500 MB')
  candidatePicked.value.push(item)
}
function confirmCandidates(){candidateGroup.downloadFiles=[...candidatePicked.value];requestKey=null;candidateOpen.value=false}
function removeCandidate(group,item){if(busy.value)return;group.downloadFiles=group.downloadFiles.filter(x=>x.file_id!==item.file_id);requestKey=null}
let groupKey=0,timer,requestKey=null,stopped=false
function uuid(){return '10000000-1000-4000-8000-100000000000'.replace(/[018]/g,c=>(Number(c)^crypto.getRandomValues(new Uint8Array(1))[0]&15>>Number(c)/4).toString(16))}
function addGroup(){groups.value.push({key:++groupKey,name:'',caption:'',count:10,files:[],localImages:[],downloadFiles:[]})}
function chooseFiles(g,event){g.files=Array.from(event.target.files);requestKey=null}
addGroup()
const totalImages=computed(()=>groups.value.reduce((n,g)=>n+(g.count||0),0))
const stages=[{key:'training',label:'训练'},{key:'generating',label:'生成'},{key:'labeling',label:'自动标注'},{key:'completed',label:'人工修正'}]
const stageIndex=computed(()=>selected.value?.status==='importing'?2:stages.findIndex(s=>s.key===selected.value?.status))
const trainPercent=computed(()=>Math.min(100,Math.round(100*(selected.value?.progress.step||0)/(selected.value?.progress.total_steps||selected.value?.options.steps||1))))
function statusText(s){return ({draft:'待提交',queued:'排队中',training:'训练中',generating:'生成中',importing:'入库中',labeling:'自动标注中',completed:'待人工修正',failed:'执行失败'})[s]||s}
async function select(id){selected.value=await api.get(`/fewshot/jobs/${id}`)}
async function refresh(){const data=await api.get('/fewshot/jobs',{params:{page:page.value,all_users:allUsers.value},silent:true});jobs.value=data.items;if(selected.value)await select(selected.value.id)}
async function changePage(delta){page.value+=delta;await refresh()}
async function submit(id){acting.value=true;try{await api.post(`/fewshot/jobs/${id}/submit`);await select(id);await refresh();ElMessage.success('任务已进入队列')}finally{acting.value=false}}
async function create(){
  if(!form.name.trim()||groups.value.some(g=>!g.name.trim()||!g.caption.trim()||!(g.files.length+g.localImages.length+g.downloadFiles.length)))return ElMessage.warning('请填写任务名称，并为每组添加图片和共用描述')
  const files=groups.value.flatMap(g=>[...g.files,...g.localImages,...g.downloadFiles])
  if(files.length>100||files.some(f=>f.size>20*1024**2)||files.reduce((n,f)=>n+f.size,0)>500*1024**2||totalImages.value>1000)return ElMessage.warning('请检查上传大小、素材数量与生成数量限制')
  busy.value=true;uploadProgress.value=0
  try{
    requestKey ||= uuid()
    const payload=new FormData()
    payload.append('manifest',JSON.stringify({...form,request_key:requestKey,options,download_request_id:downloadRequest.value?.id,groups:groups.value.map(({name,caption,count,localImages,downloadFiles})=>({name,caption,count,data_file_ids:downloadFiles.map(f=>f.file_id),local_images:localImages.map(({dataset,name})=>({dataset,name}))}))}))
    groups.value.forEach((g,i)=>g.files.forEach(f=>payload.append(`files_${i}`,f)))
    const job=await api.post('/fewshot/jobs',payload,{timeout:600000,onUploadProgress:e=>{uploadProgress.value=Math.round(100*e.loaded/(e.total||1))}})
    requestKey=null;await select(job.id);await submit(job.id)
  }finally{busy.value=false;await refresh()}
}
async function regenerate(){acting.value=true;try{const job=await api.post(`/fewshot/jobs/${selected.value.id}/generate`);await select(job.id);await refresh();ElMessage.success('已创建复用任务，确认后点击提交；数量沿用原任务，种子已更新')}finally{acting.value=false}}
async function poll(){try{await refresh()}catch{/* authentication handled by API */}if(!stopped)timer=setTimeout(poll,4000)}
onMounted(async()=>{
  capability.value=await api.get('/fewshot/capabilities')
  if(route.query.download_request_id){
    const r=await api.get(`/download/requests/${Number(route.query.download_request_id)}`)
    downloadRequest.value=r
    if(r.created_by===auth.user?.id){
      Object.assign(form,{name:`下载补充 #${r.id}`,project:r.query.project||'',scene:r.query.scene||'',secret_level:r.query.secret_level||levels[Math.max(0,...r.candidates.map(f=>levels.indexOf(f.secret_level)))]})
      Object.assign(groups.value[0],{name:(r.query.labels.join('、')||r.query.scene||'补充素材').slice(0,80),caption:r.query.labels.length?`a photo of ${r.query.labels.join(', ')}`.slice(0,2000):'',count:r.generation_count||10,downloadFiles:r.candidates})
    }
  }
  await poll()
  if(route.query.job_id){await select(Number(route.query.job_id));if(selected.value.created_by!==auth.user?.id&&auth.canViewUsers){allUsers.value=true;await refresh()}}
})
onUnmounted(()=>{stopped=true;clearTimeout(timer)})
</script>
<style scoped>
.material-toolbar{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:16px}.material-toolbar .el-select{min-width:0}.material-results{min-height:100px;max-height:55vh;overflow-y:auto;padding-right:4px}.material-section h4{color:var(--dp-text);font-size:14px;margin:6px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--dp-border)}.material-section{margin-bottom:18px}.material-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.material-tile{display:grid;grid-template-columns:110px minmax(0,1fr);gap:12px;padding:10px;border:2px solid var(--dp-border);background:var(--dp-bg);color:var(--dp-text);min-width:0;border-radius:5px}.material-tile.picked{border-color:var(--dp-primary);background:var(--dp-bg-hover)}.material-select{align-self:start;border:0;background:transparent;color:var(--dp-text);cursor:pointer;padding:0;width:100%}.material-select:focus-visible{outline:2px solid var(--dp-primary);outline-offset:3px}.material-select img{width:100%;height:100px;object-fit:contain}.material-select span{display:block;font-size:12px;margin-top:6px}.material-description{min-width:0}.material-label{display:block;font-size:13px;font-weight:500;overflow-wrap:anywhere;color:var(--dp-primary)}.material-caption{font-size:12px;line-height:1.6;margin:6px 0;max-height:140px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere}.material-filename{display:block;overflow-wrap:anywhere;font:10px ui-monospace,monospace;color:var(--dp-text-secondary);margin-top:6px}.material-tile small{font-size:10px;color:var(--dp-text-secondary)}.local-selected{display:flex;gap:5px;flex-wrap:wrap;max-height:110px;overflow:auto;margin:10px 0}.local-selected .el-tag{max-width:100%}
@media(max-width:600px){.material-grid,.material-toolbar{grid-template-columns:1fr}.material-tile{grid-template-columns:100px minmax(0,1fr)}}
.caption-label{display:block;font-size:13px;color:var(--dp-text);margin:12px 0 7px}.engine-note{margin:0 0 16px}.download-context{margin-bottom:18px}.candidate-tile{text-align:left;cursor:pointer}.candidate-tile img{width:110px;height:100px;object-fit:contain}.candidate-tile:disabled{opacity:.45;cursor:default}.candidate-tile b{overflow-wrap:anywhere}.workspace{display:grid;grid-template-columns:minmax(320px,420px) minmax(0,1fr);gap:20px;margin-top:20px;align-items:start}.composer,.task-list,.detail{padding:22px;min-width:0}h3{font-size:16px;margin:0 0 16px;color:var(--dp-text)}.section-heading,.group-heading,.submit-row,.count-row{display:flex;justify-content:space-between;align-items:center;gap:12px}.section-heading h3{margin:0}.field-pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}.field-pair .el-input-number{width:100%}
.material-group{border:1px solid var(--dp-border);border-left:3px solid var(--dp-primary);padding:16px;margin-bottom:14px;border-radius:6px;background:var(--dp-bg)}.group-heading{margin-bottom:10px;font-size:13px;color:var(--dp-text)}.count-row{font-size:13px;margin-top:12px}.file-picker{display:inline-block;position:relative;border:1px solid var(--dp-border);padding:7px 12px;border-radius:4px;cursor:pointer;margin:12px 8px 12px 0;font-size:13px;background:var(--dp-bg-card)}.file-picker input{position:absolute;inset:0;opacity:0;width:100%;cursor:pointer}.file-picker:focus-within{outline:2px solid var(--dp-primary);outline-offset:2px}.hint{font-size:12px;color:var(--dp-text-secondary);line-height:1.7}.file-names{font-size:12px;max-height:50px;overflow:auto;word-break:break-all;color:var(--dp-text-secondary);margin:0 0 10px}.advanced{margin:18px 0}.submit-row strong{font:600 24px ui-monospace,monospace;color:var(--dp-primary)}.submit-row{font-size:13px}
.task-column{display:grid;gap:20px;min-width:0}.job-row{width:100%;display:flex;justify-content:space-between;gap:12px;align-items:center;text-align:left;padding:14px 10px;border:0;border-bottom:1px solid var(--dp-border);background:var(--dp-bg-card);cursor:pointer;color:var(--dp-text)}.job-row b{display:block;word-break:break-word;font-weight:500}.job-row small{display:block;margin-top:6px;color:var(--dp-text-secondary)}.job-row.selected{background:var(--dp-bg-hover);box-shadow:inset 3px 0 var(--dp-primary)}.job-row:focus-visible{outline:2px solid var(--dp-primary)}.pipeline{display:flex;margin:24px 0;gap:4px}.pipeline>div{flex:1;border-top:3px solid var(--dp-border);padding-top:12px;font-size:12px;color:var(--dp-text-secondary)}.pipeline span{font-family:monospace;margin-right:7px}.pipeline .done{border-color:var(--dp-success);color:var(--dp-success)}.pipeline .current{border-color:var(--dp-primary);color:var(--dp-primary)}.actions{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0}
.result-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin:20px 0}.result-grid figure{margin:0}.result-grid img{width:100%;aspect-ratio:1;object-fit:contain;background:var(--dp-bg);border-radius:4px}.result-grid figcaption{font:11px ui-monospace,monospace;color:var(--dp-text-secondary);margin-top:5px}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:260px;overflow:auto;font:12px/1.6 ui-monospace,monospace;background:var(--dp-bg);padding:12px}.pagination{display:flex;justify-content:center;gap:12px;align-items:center;font-size:12px;margin-top:16px}
@media(max-width:1100px){.workspace{grid-template-columns:minmax(0,1fr)}.result-grid{grid-template-columns:repeat(4,minmax(0,1fr))}}@media(max-width:600px){.field-pair{grid-template-columns:1fr}.composer,.detail,.task-list{padding:14px}.result-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.submit-row{flex-wrap:wrap}}
</style>
<style>
.material-dialog{max-width:calc(100vw - 24px)}
@media (max-width:760px) {
  html:has(.fewshot-layout),body:has(.fewshot-layout),#app:has(.fewshot-layout){min-width:0}
  .fewshot-page.page{padding:16px}
}
</style>
